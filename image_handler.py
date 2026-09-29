import os


# ============================================================
# IMAGE CONFIGURATION
# ============================================================

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


# ============================================================
# GET IMAGE PATHS
# ============================================================

def get_image_paths(image_dir, recursive=False):
    """
    Lấy danh sách đường dẫn ảnh từ thư mục.

    Parameters
    ----------
    image_dir : str
        Thư mục chứa ảnh cần phân tích.

    recursive : bool
        False -> chỉ đọc ảnh trực tiếp trong thư mục.
        True  -> đọc cả các thư mục con.

    Returns
    -------
    image_paths : list[str]
        Danh sách đường dẫn ảnh.
    """

    # --------------------------------------------------------
    # Kiểm tra thư mục
    # --------------------------------------------------------

    if not os.path.isdir(image_dir):
        raise FileNotFoundError(
            f"Không tìm thấy thư mục ảnh:\n{image_dir}"
        )

    image_paths = []

    # --------------------------------------------------------
    # Không quét thư mục con
    # --------------------------------------------------------

    if not recursive:

        for file_name in os.listdir(image_dir):

            file_path = os.path.join(
                image_dir,
                file_name
            )

            if not os.path.isfile(file_path):
                continue

            extension = os.path.splitext(
                file_name
            )[1].lower()

            if extension in SUPPORTED_IMAGE_EXTENSIONS:
                image_paths.append(file_path)

    # --------------------------------------------------------
    # Quét cả thư mục con
    # --------------------------------------------------------

    else:

        for root, _, files in os.walk(image_dir):

            for file_name in files:

                extension = os.path.splitext(
                    file_name
                )[1].lower()

                if extension not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue

                image_paths.append(
                    os.path.join(
                        root,
                        file_name
                    )
                )

    # --------------------------------------------------------
    # Sort để thứ tự chạy ổn định
    # --------------------------------------------------------

    image_paths.sort()

    # --------------------------------------------------------
    # Không có ảnh
    # --------------------------------------------------------

    if len(image_paths) == 0:
        raise ValueError(
            f"Không tìm thấy ảnh trong thư mục:\n"
            f"{image_dir}"
        )

    return image_paths
