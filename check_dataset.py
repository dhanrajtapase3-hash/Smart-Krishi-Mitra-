from pathlib import Path
from PIL import Image

DATASET_DIR = Path(r"I:\development\Dataaset")

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

files = [
    path
    for path in DATASET_DIR.rglob("*")
    if path.is_file() and path.suffix.lower() in image_extensions
]

valid_images = 0
corrupted_images = []

for path in files:
    try:
        with Image.open(path) as image:
            image.verify()

        valid_images += 1

    except Exception:
        corrupted_images.append(path)

print()
print("===================================")
print(" Smart Krishi Mitra Dataset Check")
print("===================================")
print(f"Total images     : {len(files)}")
print(f"Valid images     : {valid_images}")
print(f"Corrupted images : {len(corrupted_images)}")

if corrupted_images:
    print()
    print("Corrupted files:")
    for path in corrupted_images:
        print(path)
else:
    print()
    print("All images are readable.")