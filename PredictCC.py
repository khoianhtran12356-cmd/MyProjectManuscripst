def test_single_image(
    model,
    image_path,
    conf_threshold=0.25,
    device=None,
):
    """
    Predict một ảnh và in toàn bộ detection.
    """

    detections = predict_image(
        model=model,
        image_path=image_path,
        conf_threshold=conf_threshold,
        device=device,
    )

    print("\n" + "=" * 60)
    print("SINGLE IMAGE PREDICTION")
    print("=" * 60)

    print(f"Image      : {image_path}")
    print(f"Detections : {len(detections)}")

    for i, detection in enumerate(
        detections,
        start=1
    ):

        print(
            f"[{i}] "
            f"Char={detection['char']} | "
            f"Conf={detection['conf']:.4f} | "
            f"Box={detection['box']}"
        )

    print("=" * 60)

    return detections

def test_stage_2(
    image_dir,
    model_path,
    device=None,
    conf_threshold=0.25,
):
    """
    Test Stage 2:

        Image folder
            ↓
        Load model
            ↓
        Get image paths
            ↓
        Predict từng ảnh
    """

    from image_handler import get_image_paths

    # --------------------------------------------------------
    # 1. Load model
    # --------------------------------------------------------

    model = load_model(
        model_path=model_path,
        device=device,
    )

    # --------------------------------------------------------
    # 2. Get images
    # --------------------------------------------------------

    image_paths = get_image_paths(
        image_dir=image_dir,
        recursive=False,
    )

    print("\n" + "=" * 60)
    print("IMAGE DATASET")
    print("=" * 60)
    print(f"Image count : {len(image_paths)}")
    print("=" * 60)

    # --------------------------------------------------------
    # 3. Predict
    # --------------------------------------------------------

    all_predictions = {}

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        detections = predict_image(
            model=model,
            image_path=image_path,
            conf_threshold=conf_threshold,
            device=device,
        )

        all_predictions[image_path] = detections

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path}"
        )

        for detection in detections:

            print(
                f"    "
                f"{detection['char']} | "
                f"{detection['conf']:.3f} | "
                f"{detection['box']}"
            )

    return model, image_paths, all_predictions

