from pathlib import Path
import random
import shutil

# Original dataset
SOURCE_DIR = Path(r"I:\development\Dataaset")

# New dataset
OUTPUT_DIR = Path(r"I:\development\SmartKrishiDataset")

# Dataset split
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducible random split
RANDOM_SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

random.seed(RANDOM_SEED)


def get_images(folder):
    return [
        path
        for path in folder.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_images(images, destination):
    destination.mkdir(parents=True, exist_ok=True)

    for image in images:
        shutil.copy2(image, destination / image.name)


def main():
    if not SOURCE_DIR.exists():
        print(f"ERROR: Dataset not found: {SOURCE_DIR}")
        return

    if OUTPUT_DIR.exists():
        print(f"ERROR: Output directory already exists:")
        print(OUTPUT_DIR)
        print()
        print("Delete it manually only if you want to create the split again.")
        return

    classes = [
        folder
        for folder in SOURCE_DIR.iterdir()
        if folder.is_dir()
    ]

    if not classes:
        print("ERROR: No class folders found.")
        return

    print("======================================")
    print(" Smart Krishi Mitra Dataset Split")
    print("======================================")
    print()

    total_images = 0

    for class_folder in sorted(classes):
        class_name = class_folder.name

        images = get_images(class_folder)

        random.shuffle(images)

        total = len(images)

        train_count = int(total * TRAIN_RATIO)
        validation_count = int(total * VALIDATION_RATIO)

        train_images = images[:train_count]

        validation_images = images[
            train_count:
            train_count + validation_count
        ]

        test_images = images[
            train_count + validation_count:
        ]

        copy_images(
            train_images,
            OUTPUT_DIR / "train" / class_name,
        )

        copy_images(
            validation_images,
            OUTPUT_DIR / "validation" / class_name,
        )

        copy_images(
            test_images,
            OUTPUT_DIR / "test" / class_name,
        )

        print(
            f"{class_name:10} | "
            f"Total: {total:4} | "
            f"Train: {len(train_images):4} | "
            f"Validation: {len(validation_images):4} | "
            f"Test: {len(test_images):4}"
        )

        total_images += total

    print()
    print("--------------------------------------")
    print(f"Total images processed: {total_images}")
    print(f"Output directory      : {OUTPUT_DIR}")
    print("--------------------------------------")
    print()
    print("Dataset split completed successfully.")


if __name__ == "__main__":
    main()