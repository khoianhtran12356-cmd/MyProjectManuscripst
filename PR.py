def predict_image(
    model,
    image_path,
    conf_threshold=0.25,
    device=None,
):
    """
    Predict một ảnh và chuẩn hóa kết quả.

    Returns
    -------
    detections : list of dict

    Ví dụ:

    [
        {
            "char": "7",
            "conf": 0.96,
            "box": [100, 50, 150, 120]
        },
        {
            "char": "4",
            "conf": 0.94,
            "box": [160, 50, 210, 120]
        }
    ]
    """

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    results = model.predict(
        source=image_path,
        conf=conf_threshold,
        device=device,
        verbose=False,
    )

    # Một ảnh -> lấy result đầu tiên
    result = results[0]

    detections = []

    # --------------------------------------------------------
    # Không có detection
    # --------------------------------------------------------

    if result.boxes is None:
        return detections

    if len(result.boxes) == 0:
        return detections

    # --------------------------------------------------------
    # Class names
    # --------------------------------------------------------

    names = result.names

    # --------------------------------------------------------
    # Lấy dữ liệu tensor
    # --------------------------------------------------------

    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    # --------------------------------------------------------
    # Chuẩn hóa từng detection
    # --------------------------------------------------------

    for box, conf, cls_id in zip(
        boxes,
        confidences,
        classes
    ):

        x1, y1, x2, y2 = box

        cls_id = int(cls_id)

        # Tên class
        char = names[cls_id]

        detections.append({
            "char": str(char),
            "conf": float(conf),
            "box": [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
            ],
        })

    return detections
