from ultralytics import YOLO
import torch


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path, device=None):
    """
    Load YOLO model.

    Parameters
    ----------
    model_path : str
        Đường dẫn tới file weights.

    device : str/int/None
        Ví dụ:
            "cpu"
            "0"
            0
        Nếu None -> tự chọn CUDA nếu có.

    Returns
    -------
    model
        YOLO model.
    """

    # --------------------------------------------------------
    # Kiểm tra CUDA
    # --------------------------------------------------------

    if device is None:

        device = (
            "0"
            if torch.cuda.is_available()
            else "cpu"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = YOLO(model_path)

    # --------------------------------------------------------
    # Move model tới device
    # --------------------------------------------------------

    model.to(device)

    print("=" * 60)
    print("MODEL LOADED")
    print("=" * 60)
    print(f"Model  : {model_path}")
    print(f"Device : {device}")
    print("=" * 60)

    return model
