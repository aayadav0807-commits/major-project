
from pathlib import Path
from PIL import Image

DATASET_DIR = Path(
    "data/DyslexiaScreening/Complete And Balance Hand-written Dataset/Dataset"
)

print("=" * 50)
print("DYSLEXIA DATASET CHECK")
print("=" * 50)

total_images = 0
total_bad = 0

for class_name in ("Yes", "No"):
    class_dir = DATASET_DIR / class_name

    if not class_dir.is_dir():
        print(f"ERROR: Missing folder: {class_dir}")
        continue

    files = [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    valid = 0
    bad = 0

    for path in files:
        try:
            with Image.open(path) as img:
                img.verify()
            valid += 1
        except Exception:
            bad += 1

    total_images += len(files)
    total_bad += bad

    print(f"\nClass: {class_name}")
    print(f"Total images: {len(files)}")
    print(f"Readable images: {valid}")
    print(f"Unreadable images: {bad}")

print("\n" + "=" * 50)
print(f"Total images: {total_images}")
print(f"Total unreadable images: {total_bad}")
print("=" * 50)
