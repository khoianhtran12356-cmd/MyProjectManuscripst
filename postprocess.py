# ============================================================
# POST-PROCESSING
# ============================================================


def box_area(box):
    """
    Tính diện tích bounding box.

    box = [x1, y1, x2, y2]
    """

    x1, y1, x2, y2 = box

    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)

    return width * height


def intersection_area(box_a, box_b):
    """
    Tính diện tích phần giao nhau của 2 bounding box.
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)

    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    width = max(0.0, ix2 - ix1)
    height = max(0.0, iy2 - iy1)

    return width * height


def overlap_ratio(box_a, box_b):
    """
    Tính mức độ chồng chập dựa trên diện tích box nhỏ hơn.

    overlap =
        intersection_area / min(area_a, area_b)

    Ví dụ:
        Box B nằm hoàn toàn trong Box A
        -> overlap gần 100%

    Hai box chỉ chạm cạnh
        -> overlap = 0
    """

    area_a = box_area(box_a)
    area_b = box_area(box_b)

    if area_a <= 0 or area_b <= 0:
        return 0.0

    intersection = intersection_area(
        box_a,
        box_b
    )

    return intersection / min(
        area_a,
        area_b
    )


def filter_overlapping_boxes(
    detections,
    overlap_threshold=0.50
):
    """
    Loại các prediction chồng chập quá mức.

    Nếu 2 box overlap > 50%:
        -> giữ box có confidence cao hơn.

    Nếu chỉ chạm cạnh:
        -> không loại.

    Parameters
    ----------
    detections : list
        Danh sách detection từ Stage 2.

    overlap_threshold : float
        Ngưỡng overlap.

    Returns
    -------
    filtered : list
        Danh sách detection sau filtering.
    """

    if not detections:
        return []

    # --------------------------------------------------------
    # Sort confidence giảm dần
    # --------------------------------------------------------

    candidates = sorted(
        detections,
        key=lambda x: x["conf"],
        reverse=True
    )

    filtered = []

    while candidates:

        # Box confidence cao nhất
        current = candidates.pop(0)

        filtered.append(current)

        remaining = []

        for candidate in candidates:

            overlap = overlap_ratio(
                current["box"],
                candidate["box"]
            )

            # ------------------------------------------------
            # Nếu overlap > 50%
            # giữ current vì confidence cao hơn
            # ------------------------------------------------

            if overlap > overlap_threshold:
                continue

            remaining.append(candidate)

        candidates = remaining

    return filtered


def validate_detections(
    detections,
    expected_chars=2
):
    """
    Kiểm tra prediction sau filtering.

    Yêu cầu:
        - chính xác 2 ký tự
    """

    if len(detections) != expected_chars:

        return False, (
            f"Expected {expected_chars} characters, "
            f"but got {len(detections)}"
        )

    return True, None


def get_center_x(detection):
    """
    Lấy tọa độ tâm X của bounding box.
    """

    x1, _, x2, _ = detection["box"]

    return (x1 + x2) / 2.0


def build_cc_code(detections):
    """
    Sắp xếp 2 ký tự từ trái sang phải
    và ghép thành CC code.

    Ví dụ:

        7      4
        ↓      ↓
       Char1  Char2

        -> "74"
    """

    if len(detections) != 2:
        raise ValueError(
            "build_cc_code() yêu cầu đúng 2 detections."
        )

    # --------------------------------------------------------
    # Sort trái -> phải
    # --------------------------------------------------------

    sorted_detections = sorted(
        detections,
        key=get_center_x
    )

    char1 = str(
        sorted_detections[0]["char"]
    ).strip().upper()

    char2 = str(
        sorted_detections[1]["char"]
    ).strip().upper()

    cc_code = char1 + char2

    return cc_code, sorted_detections


def process_detections(
    detections,
    overlap_threshold=0.50
):
    """
    Pipeline hoàn chỉnh cho prediction của một ảnh.

        Raw detections
              ↓
        overlap filtering
              ↓
        kiểm tra đúng 2
              ↓
        sort theo X
              ↓
        ghép CC
    """

    # --------------------------------------------------------
    # Step 1: Filter overlap
    # --------------------------------------------------------

    filtered = filter_overlapping_boxes(
        detections,
        overlap_threshold=overlap_threshold
    )

    # --------------------------------------------------------
    # Step 2: Validate
    # --------------------------------------------------------

    valid, reason = validate_detections(
        filtered
    )

    if not valid:

        return {
            "success": False,
            "cc_code": None,
            "detections": filtered,
            "reason": reason,
        }

    # --------------------------------------------------------
    # Step 3: Build CC
    # --------------------------------------------------------

    cc_code, sorted_detections = build_cc_code(
        filtered
    )

    return {
        "success": True,
        "cc_code": cc_code,
        "detections": sorted_detections,
        "reason": None,
    }
