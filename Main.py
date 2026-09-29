from model_handler import test_stage_2


if __name__ == "__main__":

    image_dir = r"E:\Data_KHOI\CC_Project\Images"

    model_path = r"E:\Data_KHOI\CC_Project\Models\best.pt"

    model, image_paths, all_predictions = test_stage_2(
        image_dir=image_dir,
        model_path=model_path,
        device="0",
        conf_threshold=0.25,
    )
