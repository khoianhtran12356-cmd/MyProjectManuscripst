import os

from excel_handler import (
    load_excel_template,
    read_cc_codes,
    validate_cc_codes,
    build_cc_lookup,
    update_excel_counts,
    save_excel_template,
)

from image_handler import (
    get_image_paths,
)

from model_handler import (
    load_model,
)

from analyzer import (
    analyze_dataset,
    save_analysis_log,
)


def main(
    image_dir,
    excel_template,
    model_path,
    output_excel,
    output_log,
    sheet_name=None,
    device="0",
    conf_threshold=0.25,
    overlap_threshold=0.50,
    reset_before_update=True,
):
    """
    COMPLETE CC CODE ANALYSIS PIPELINE

        Excel template
              +
        Image dataset
              +
        Trained model
              ↓
        CC Code Result Excel
    """

    print("\n")
    print("=" * 70)
    print("CC CODE ANALYSIS TOOL")
    print("=" * 70)

    # ========================================================
    # STAGE 1
    # EXCEL TEMPLATE
    # ========================================================

    print("\n[STAGE 1] Loading Excel template...")

    wb, ws = load_excel_template(
        excel_template,
        sheet_name=sheet_name,
    )

    records = read_cc_codes(ws)

    validate_cc_codes(records)

    cc_lookup = build_cc_lookup(records)

    print(
        f"Template loaded successfully: "
        f"{len(cc_lookup)} CC codes"
    )

    # ========================================================
    # STAGE 2
    # IMAGE + MODEL
    # ========================================================

    print("\n[STAGE 2] Loading image dataset...")

    image_paths = get_image_paths(
        image_dir=image_dir,
        recursive=False,
    )

    print(
        f"Found {len(image_paths)} images."
    )

    print("\n[STAGE 2] Loading model...")

    model = load_model(
        model_path=model_path,
        device=device,
    )

    # ========================================================
    # STAGE 3 + 4
    # PREDICT + POST PROCESS + CC ANALYSIS
    # ========================================================

    print(
        "\n[STAGE 3-4] "
        "Analyzing CC codes..."
    )

    cc_counter, stats, logs = analyze_dataset(
        image_paths=image_paths,
        model=model,
        cc_lookup=cc_lookup,
        conf_threshold=conf_threshold,
        overlap_threshold=overlap_threshold,
        device=device,
    )

    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("ANALYSIS SUMMARY")
    print("=" * 70)

    print(
        f"Total images           : "
        f"{stats['total_images']}"
    )

    print(
        f"Successful detection   : "
        f"{stats['successful_detection']}"
    )

    print(
        f"Invalid detection      : "
        f"{stats['invalid_detection']}"
    )

    print(
        f"Valid CC                : "
        f"{stats['valid_cc']}"
    )

    print(
        f"Unknown CC              : "
        f"{stats['unknown_cc']}"
    )

    print(
        f"Prediction errors       : "
        f"{stats['prediction_error']}"
    )

    print("\nCC COUNTS")
    print("-" * 40)

    for cc_code in sorted(cc_counter):

        print(
            f"{cc_code:>5} : "
            f"{cc_counter[cc_code]}"
        )

    print("=" * 70)

    # ========================================================
    # STAGE 5
    # UPDATE EXCEL
    # ========================================================

    print(
        "\n[STAGE 5] Updating Excel..."
    )

    update_excel_counts(
        ws=ws,
        cc_lookup=cc_lookup,
        cc_counter=cc_counter,
        reset_before_update=reset_before_update,
    )

    # ========================================================
    # SAVE EXCEL
    # ========================================================

    save_excel_template(
        wb=wb,
        output_path=output_excel,
    )

    # ========================================================
    # SAVE LOG
    # ========================================================

    save_analysis_log(
        logs=logs,
        output_csv=output_log,
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print("\n")
    print("=" * 70)
    print("PROCESS COMPLETED SUCCESSFULLY")
    print("=" * 70)

    return {
        "stats": stats,
        "cc_counter": dict(cc_counter),
        "logs": logs,
        "output_excel": output_excel,
        "output_log": output_log,
    }

if __name__ == "__main__":

    image_dir = (
        r"E:\Data_KHOI\CC_Project\Images"
    )

    excel_template = (
        r"E:\Data_KHOI\CC_Project\Template"
        r"\CC_Template.xlsm"
    )

    model_path = (
        r"E:\Data_KHOI\CC_Project\Models"
        r"\best.pt"
    )

    output_excel = (
        r"E:\Data_KHOI\CC_Project\Result"
        r"\CC_Result.xlsm"
    )

    output_log = (
        r"E:\Data_KHOI\CC_Project\Result"
        r"\CC_Analysis_Log.csv"
    )

    main(
        image_dir=image_dir,
        excel_template=excel_template,
        model_path=model_path,

        output_excel=output_excel,
        output_log=output_log,

        device="0",

        conf_threshold=0.25,

        overlap_threshold=0.50,

        # True = kết quả của dataset hiện tại
        # False = cộng vào Repeat cũ
        reset_before_update=True,
    )
