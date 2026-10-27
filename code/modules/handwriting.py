from pathlib import Path
from PIL import Image


# Location of the handwriting dataset
DATASET_PATH = Path("data/Gambo/Train")


def check_dataset():
    classes = ["Corrected", "Normal", "Reversal"]

    print("\nHandwriting Dataset Check")
    print("-" * 40)

    total_images = 0

    for class_name in classes:
        class_path = DATASET_PATH / class_name

        if not class_path.exists():
            print(f"{class_name}: Folder not found")
            continue

        images = list(class_path.glob("*.png"))

        valid_images = 0

        for image_path in images:
            try:
                with Image.open(image_path) as img:
                    img.verify()
                valid_images += 1
            except Exception:
                pass

        print(f"{class_name}: {valid_images} valid PNG images")
        total_images += valid_images

    print("-" * 40)
    print(f"Total valid images: {total_images}")


if __name__ == "__main__":
    check_dataset()