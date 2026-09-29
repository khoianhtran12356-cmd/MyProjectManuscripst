from collections import Counter
from postprocess import process_detections


def match_cc_code(cc_code, cc_lookup):
    """
    Kiểm tra CC code có nằm trong template hay không.

    Returns
    -------
    dict or None
    """

    return cc_lookup.get(cc_code)


def analyze_dataset(
    image_paths,
    model,
    cc_lookup,
    conf_threshold=0.25,
    overlap_threshold=0.50,
    device=None,
):
    """
    Phân tích toàn bộ dataset.

    Pipeline:

        image
          ↓
        predict
          ↓
        filter
          ↓
        exactly 2 chars
          ↓
        build CC
          ↓
        match template
          ↓
        counter
    """

    # Import tại đây để tránh circular import
    from model_handler import predict_image

    # --------------------------------------------------------
    # Counter
    # --------------------------------------------------------

    cc_counter = Counter()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = {
        "total_images": 0,
        "successful_detection": 0,
        "invalid_detection": 0,
        "valid_cc": 0,
        "unknown_cc": 0,
        "prediction_error": 0,
    }

    # --------------------------------------------------------
    # Logs
    # --------------------------------------------------------

    logs = []

    # --------------------------------------------------------
    # Process images
    # --------------------------------------------------------

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        stats["total_images"] += 1

        file_name = image_path.split("\\")[-1]

        print(
            f"[{index}/{len(image_paths)}] "
            f"{file_name}",
            end=" -> "
        )

        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        try:

            detections = predict_image(
                model=model,
                image_path=image_path,
                conf_threshold=conf_threshold,
                device=device,
            )

        except Exception as error:

            stats["prediction_error"] += 1

            logs.append({
                "image": image_path,
                "status": "PREDICTION_ERROR",
                "cc_code": "",
                "reason": str(error),
            })

            print("PREDICTION ERROR")

            continue

        # ----------------------------------------------------
        # Post-process
        # ----------------------------------------------------

        result = process_detections(
            detections,
            overlap_threshold=overlap_threshold,
        )

        # ----------------------------------------------------
        # Invalid detection
        # ----------------------------------------------------

        if not result["success"]:

            stats["invalid_detection"] += 1

            logs.append({
                "image": image_path,
                "status": "INVALID_DETECTION",
                "cc_code": "",
                "reason": result["reason"],
            })

            print(
                f"INVALID - {result['reason']}"
            )

            continue

        # ----------------------------------------------------
        # CC code
        # ----------------------------------------------------

        cc_code = result["cc_code"]

        stats["successful_detection"] += 1

        # ----------------------------------------------------
        # Match template
        # ----------------------------------------------------

        cc_info = match_cc_code(
            cc_code,
            cc_lookup
        )

        if cc_info is None:

            stats["unknown_cc"] += 1

            logs.append({
                "image": image_path,
                "status": "UNKNOWN_CC",
                "cc_code": cc_code,
                "reason": (
                    "CC code không tồn tại "
                    "trong template."
                ),
            })

            print(
                f"{cc_code} -> UNKNOWN"
            )

            continue

        # ----------------------------------------------------
        # Valid CC
        # ----------------------------------------------------

        cc_counter[cc_code] += 1

        stats["valid_cc"] += 1

        logs.append({
            "image": image_path,
            "status": "OK",
            "cc_code": cc_code,
            "reason": "",
        })

        print(
            f"{cc_code} -> OK"
        )

    return cc_counter, stats, logs
