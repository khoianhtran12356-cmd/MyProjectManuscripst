"""Mô phỏng Shiftwise bằng PyTorch thuần: group conv 3x3 -> dời bằng kernel one-hot -> crop -> sum K.
Chỉ xét kernel tương đương M x 3 (axis='h') hoặc 3 x M (axis='w'), M lẻ >= 7."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def shift_pad_crop(z, d, dim):
    """Bản 'phép cơ bản' của một độ dời: out[p] = z[p + d], ngoài biên = 0, giữ nguyên kích thước.
    d >= 0: cắt d phần tử đầu rồi đệm 0 ở cuối; d < 0: cắt cuối rồi đệm 0 ở đầu."""
    L = z.shape[dim]
    if abs(d) >= L:
        return torch.zeros_like(z)
    zshape = list(z.shape); zshape[dim] = abs(d)
    zero = z.new_zeros(zshape)
    if d >= 0:
        return torch.cat([z.narrow(dim, d, L - d), zero], dim)
    return torch.cat([zero, z.narrow(dim, 0, L + d)], dim)


class ShiftWiseStrip(nn.Module):
    N = 3  # kernel nhỏ cố định 3x3, extra_pad = N-1-N//2 = 1

    def __init__(self, C, M, axis='h', shuffle=False, exact_taps=False, seed=0):
        super().__init__()
        assert M % 2 == 1 and M >= 7 and axis in ('h', 'w')
        self.C, self.M, self.axis = C, M, axis
        self.K = K = math.ceil(M / 3)
        self.mid = mid = M // 2

        # block (vị trí cửa sổ trên trục kernel lớn) của lát k ở kênh vào c
        g = torch.Generator().manual_seed(seed)
        if shuffle:
            perm = torch.stack([torch.randperm(K, generator=g) for _ in range(C)])
        else:
            perm = torch.arange(K).repeat(C, 1)
        self.register_buffer('perm', perm)                           # (C, K)

        # tâm cửa sổ = độ dời d của lát:  c = -mid + 3*block + 1
        centers = (-mid + 3 * perm + 1).reshape(-1)                  # (C*K,)
        cmax = int((-mid + 3 * torch.arange(K) + 1).abs().max())
        self.S = S = 2 * cmax + 1                                    # kích thước kernel one-hot
        oh = torch.zeros(C * K, 1, S)
        oh[torch.arange(C * K), 0, S // 2 + centers] = 1.0           # out[p] = in[p + d]
        self.register_buffer('onehot', oh.view(C * K, 1, 1, S) if axis == 'w' else oh.view(C * K, 1, S, 1))

        # (tuỳ chọn) khoá các tap vượt quá M của cửa sổ cuối khi M không chia hết cho 3
        tap = -mid + 3 * perm.reshape(-1, 1) + torch.arange(3)       # (C*K, 3)
        keep = (tap <= mid).float() if exact_taps else torch.ones(C * K, 3)
        self.register_buffer('mask', keep.view(C * K, 1, 1, 3) if axis == 'w' else keep.view(C * K, 1, 3, 1))

        self.conv = nn.Conv2d(C, C * K, 3, padding=2, groups=C, bias=False)  # padding N-1 => lề dư 1 mỗi phía

    def forward(self, x):
        B, C, H, W = x.shape
        K, S = self.K, self.S
        z = F.conv2d(x, self.conv.weight * self.mask, padding=2, groups=C)       # (B, C*K, H+2, W+2)
        pad = (0, S // 2) if self.axis == 'w' else (S // 2, 0)
        z = F.conv2d(z, self.onehot, padding=pad, groups=C * K)                  # dời, đệm 0, giữ H+2, W+2
        z = z[:, :, 1:-1, 1:-1]                                                  # crop extra_pad -> H, W
        return z.reshape(B, C, K, H, W).sum(2)                                   # sum K lát -> C kênh

    # ---- kiểm chứng ----
    @torch.no_grad()
    def equivalent_weight(self):
        C, K = self.C, self.K
        w = (self.conv.weight * self.mask).view(C, K, 3, 3)
        wb = w.new_zeros((C, 1, 3, 3 * K) if self.axis == 'w' else (C, 1, 3 * K, 3))
        for c in range(C):
            for k in range(K):
                b = int(self.perm[c, k])
                if self.axis == 'w':
                    wb[c, 0, :, 3 * b:3 * b + 3] = w[c, k]
                else:
                    wb[c, 0, 3 * b:3 * b + 3, :] = w[c, k]
        return wb

    def reference(self, x):
        """Conv depthwise kernel lớn tương đương (3K tap, tap đầu ở -mid)."""
        wb, r = self.equivalent_weight(), 3 * self.K - 1 - self.mid
        xp = F.pad(x, (self.mid, r, 1, 1)) if self.axis == 'w' else F.pad(x, (1, 1, self.mid, r))
        return F.conv2d(xp, wb, groups=self.C)


class ShiftWiseBlock(nn.Module):
    """3 nhánh như Fig.1: strip ngang + strip dọc + 3x3, mỗi nhánh BN rồi cộng (không chia sẻ group conv)."""
    def __init__(self, C, M, **kw):
        super().__init__()
        self.h = ShiftWiseStrip(C, M, 'h', **kw)
        self.w = ShiftWiseStrip(C, M, 'w', **kw)
        self.s = nn.Conv2d(C, C, 3, padding=1, groups=C, bias=False)
        self.bn = nn.ModuleList(nn.BatchNorm2d(C) for _ in range(3))

    def forward(self, x):
        return self.bn[0](self.h(x)) + self.bn[1](self.w(x)) + self.bn[2](self.s(x))


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(2, 4, 20, 17)           # H, W nhỏ hơn dịch lớn nhất để thử zero-fill
    for axis in 'hw':
        for M in (7, 9, 11, 13, 31):
            for sh in (False, True):
                m = ShiftWiseStrip(4, M, axis, shuffle=sh)
                err = (m(x) - m.reference(x)).abs().max().item()
                assert err < 1e-5, (axis, M, sh, err)
    # exact_taps: bằng đúng conv same M tap (padding M//2, 1)
    m = ShiftWiseStrip(4, 13, 'w', exact_taps=True)
    ref = F.conv2d(x, m.equivalent_weight()[..., :13], padding=(1, 6), groups=4)
    assert (m(x) - ref).abs().max() < 1e-5
    m = ShiftWiseStrip(4, 13, 'h', exact_taps=True)
    ref = F.conv2d(x, m.equivalent_weight()[:, :, :13, :], padding=(6, 1), groups=4)
    assert (m(x) - ref).abs().max() < 1e-5
    # shift_pad_crop khớp one-hot
    z = torch.randn(1, 1, 5, 12)
    for d in (-4, -1, 0, 3, 11, 20):
        S = 41; k = torch.zeros(1, 1, 1, S); k[0, 0, 0, S // 2 + d] = 1 if abs(d) <= 20 else 0
        assert torch.allclose(F.conv2d(z, k, padding=(0, S // 2)), shift_pad_crop(z, d, 3))
    print(ShiftWiseBlock(8, 13)(torch.randn(2, 8, 16, 16)).shape, 'OK')
