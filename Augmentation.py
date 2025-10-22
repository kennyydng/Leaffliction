#!/usr/bin/env python3
import sys
import os
from PIL import Image
import random
from pathlib import Path
from collections import defaultdict

# Global constants
AUGMENTATION_TYPES = {
    "flip": lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
    "rotate": lambda img: img.rotate(random.uniform(-30, 30), expand=True),
    "skew": lambda img: img.transform(
        img.size,
        Image.PERSPECTIVE,
        (1, 0.2, 0, 0, 1, 0, 0, 0.0008),
        Image.BICUBIC
    ),
    "shear": lambda img: img.transform(
        img.size,
        Image.AFFINE,
        (1, random.uniform(-0.3, 0.3), 0, 0, 1, 0),
        Image.BICUBIC,
    ),
    "crop": lambda img: img.crop(
        (
            random.randint(0, img.size[0] - int(min(img.size) * 0.8)),
            random.randint(0, img.size[1] - int(min(img.size) * 0.8)),
            random.randint(0, img.size[0] - int(min(img.size) * 0.8))
            + int(min(img.size) * 0.8),
            random.randint(0, img.size[1] - int(min(img.size) * 0.8))
            + int(min(img.size) * 0.8),
        )
    ),
    "distort": lambda img: img.resize(
        (
            int(img.size[0] * random.uniform(0.8, 1.2)),
            int(img.size[1] * random.uniform(0.8, 1.2)),
        ),
        Image.BICUBIC,
    ),
}

IMAGE_PATTERNS = ["*.jpg", "*.jpeg", "*.JPG", "*.JPEG"]


def apply_augmentation(img_path, aug_name=None):
    """Apply one or all augmentations to an image."""
    img = Image.open(img_path)
    path = Path(img_path)

    if aug_name:
        if aug_name not in AUGMENTATION_TYPES:
            raise ValueError(f"Unknown augmentation: {aug_name}")
        augmented = AUGMENTATION_TYPES[aug_name](img)
        output_path = path.parent / f"{path.stem}_{aug_name}{path.suffix}"
        augmented.save(output_path)
        return 1

    # Apply all augmentations
    for name, func in AUGMENTATION_TYPES.items():
        output_path = path.parent / f"{path.stem}_{name}{path.suffix}"
        func(img).save(output_path)
        print(f"Saved {output_path}")
    return len(AUGMENTATION_TYPES)


def is_original(img_path):
    """Check if image is original (not augmented)."""
    return not any(f"_{aug}" in img_path.stem for aug in AUGMENTATION_TYPES)


def get_class_info(directory_path):
    """Get image counts and paths per class."""
    directory = Path(directory_path)
    class_counts = defaultdict(int)
    class_paths = defaultdict(list)

    for pattern in IMAGE_PATTERNS:
        for img_path in directory.rglob(pattern):
            if is_original(img_path):
                class_name = img_path.parent.name
                class_counts[class_name] += 1
                class_paths[class_name].append(img_path)

    return class_counts, class_paths


def process_directory(directory_path, target_count):
    """Process directory to reach target count per class."""
    class_counts, class_paths = get_class_info(directory_path)

    print("\nCurrent class distribution:")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count}")

    total_generated = 0
    for class_name, count in class_counts.items():
        if count >= target_count:
            print(f"\n{class_name}: {count} >= {target_count}")
            continue

        needed = target_count - count
        print(f"\n{class_name}: Generating {needed} additional images")
        generated = 0

        for img_path in class_paths[class_name]:
            if generated >= needed:
                break

            try:
                img = Image.open(img_path)
                for aug_name in AUGMENTATION_TYPES:
                    if generated >= needed:
                        break
                    output_path = (
                        img_path.parent /
                        f"{img_path.stem}_{aug_name}{img_path.suffix}"
                    )
                    AUGMENTATION_TYPES[aug_name](img).save(output_path)
                    print(f"Saved {output_path}")
                    generated += 1
                    total_generated += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    return total_generated


def validate_path(path: Path) -> None:
    """Validate path existence and permissions."""
    if not path.exists():
        raise ValueError(f"Path '{path}' does not exist")

    if not os.access(path, os.R_OK):
        raise PermissionError(f"No read permission for '{path}'")

    if path.is_dir():
        if not os.access(path, os.W_OK):
            raise PermissionError(f"No write permission in directory '{path}'")


def main():
    if len(sys.argv) < 2:
        print("Usage: ./Augmentation.py <path> [--target N]")
        print("Examples:")
        print("./Augmentation.py ./input/Apple --target 1640")
        print("./Augmentation.py image.jpg")
        sys.exit(1)

    # Si c'est un dossier, vérifier qu'on a bien l'argument --target
    path = Path(sys.argv[1])
    if path.is_dir() and (len(sys.argv) < 4 or "--target" not in sys.argv):
        print("Error: --target argument required for directory processing")
        print("Example: ./Augmentation.py ./input/Apple --target 1640")
        sys.exit(1)

    try:
        validate_path(path)
        print(f"Processing: {path}")

        if path.is_file():
            # Validate file type
            if not path.suffix.lower() in [".jpg", ".jpeg"]:
                raise ValueError(f"Unsupported file type: {path.suffix}")

            if not is_original(path):
                raise ValueError(f"'{path}' appears to be already augmented")

            try:
                count = apply_augmentation(path)
                print(f"Done: {count} augmentations applied")
            except Exception as e:
                raise RuntimeError(f"Failed to process {path}: {str(e)}")

        else:
            # Get target value from command line arguments
            try:
                target_idx = sys.argv.index('--target') + 1
                target_count = int(sys.argv[target_idx])
            except (ValueError, IndexError):
                raise ValueError(
                    "--target argument required for directory processing\n"
                    "Example: ./Augmentation.py ./input/Apple --target 1640"
                )

            if target_count <= 0:
                raise ValueError("Target count must be positive")

            if target_count > 10000:
                raise ValueError("Target count too high (max: 10000)")

            print(f"Scanning for images in: {path}")
            try:
                count = process_directory(path, target_count)
                print(f"\nDone: {count} new images generated")
            except Exception as e:
                raise RuntimeError(f"Failed to process directory: {str(e)}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
