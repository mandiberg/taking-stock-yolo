from pathlib import Path

import cv2


FOLDER = Path(
    "/Users/michaelmandiberg/Library/CloudStorage/Dropbox/Mandiberg-Heft/book_assets/spreads_jpg"
)
OUTPUT_FOLDER_NAME = "split_pages"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
OUTPUT_EXTENSION = ".jpg"


def split_spread(image_path: Path, output_dir: Path, page_number: int) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return 0

    height, width = image.shape[:2]
    midpoint = width // 2

    left_half = image[:, :midpoint]
    right_half = image[:, midpoint:]

    left_output = output_dir / f"{page_number}{OUTPUT_EXTENSION}"
    right_output = output_dir / f"{page_number + 1}{OUTPUT_EXTENSION}"

    cv2.imwrite(str(left_output), left_half)
    cv2.imwrite(str(right_output), right_half)

    print(f"Saved {left_output.name} and {right_output.name} from {image_path.name}")
    return 2


def main() -> None:
    if not FOLDER.exists():
        raise FileNotFoundError(f"Folder not found: {FOLDER}")

    output_dir = FOLDER / OUTPUT_FOLDER_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in FOLDER.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_paths:
        print(f"No supported images found in {FOLDER}")
        return

    page_number = 1
    for image_path in image_paths:
        pages_written = split_spread(image_path, output_dir, page_number)
        page_number += pages_written

    print(f"Done. Split files saved to: {output_dir}")


if __name__ == "__main__":
    main()