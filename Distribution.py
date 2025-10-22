#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Configure TensorFlow logging before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Matplotlib configuration
plt.rcParams.update(
    {
        "figure.autolayout": True,
        "axes.labelweight": "bold",
        "axes.labelsize": "large",
        "axes.titleweight": "bold",
        "axes.titlesize": 18,
        "axes.titlepad": 10,
    }
)
plt.ion()  # Enable interactive mode

# Constants
IMAGE_SIZE = [128, 128]  # Standard size for all images


def validate_directory(directory: str) -> None:
    """
    Validate directory structure and content.

    Args:
        directory (str): Path to check

    Raises:
        ValueError: If directory structure is invalid
        PermissionError: If permissions are insufficient
    """
    path = Path(directory)

    if not path.exists():
        raise ValueError(f"Directory '{directory}' does not exist")

    if not path.is_dir():
        raise ValueError(f"'{directory}' is not a directory")

    if not os.access(directory, os.R_OK):
        raise PermissionError(f"No read permission for '{directory}'")

    # Check for subdirectories (classes)
    subdirs = [x for x in path.iterdir() if x.is_dir()]
    if not subdirs:
        raise ValueError(f"No class subdirectories found in '{directory}'")

    # Check for images in subdirectories
    valid_extensions = {".jpg", ".jpeg", ".png"}
    for subdir in subdirs:
        has_images = False
        for ext in valid_extensions:
            if list(subdir.glob(f"*{ext}")):
                has_images = True
                break
        if not has_images:
            raise ValueError(f"No valid images found in '{subdir}'")


def get_dataset_info(directory: str) -> dict:
    """
    Analyze image distribution in the given directory.

    Args:
        directory (str): Path to the dataset directory

    Returns:
        dict: Category names as keys and image counts as values

    Raises:
        ValueError: If directory structure or content is invalid
        RuntimeError: If dataset processing fails
    """
    try:
        # Validate directory structure first
        validate_directory(directory)

        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="categorical",
            image_size=IMAGE_SIZE,
            shuffle=False,
            batch_size=None,
            validation_split=None,
            seed=None,
            color_mode="rgb",
        )

        class_names = dataset.class_names
        image_counts = {}

        # Count images per class
        for images, labels in dataset:
            class_idx = np.argmax(labels.numpy())
            class_name = class_names[class_idx]
            image_counts[class_name] = image_counts.get(class_name, 0) + 1

        if not image_counts:
            raise ValueError("No valid images found in the dataset")

        return image_counts

    except tf.errors.OpError as e:
        raise RuntimeError(f"TensorFlow error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Failed to process dataset: {str(e)}")


def create_charts(data: dict, title: str) -> None:
    """
    Create and save visualization charts for the dataset.

    Args:
        data (dict): Category names as keys and image counts as values
        title (str): Title for the charts (usually dataset name)
    """
    # Créer une figure avec deux sous-plots côte à côte
    plt.figure(figsize=(20, 8))

    # Configuration des couleurs
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(data)))

    # Graphique en camembert (à gauche)
    plt.subplot(1, 2, 1)
    wedges, texts, autotexts = plt.pie(
        data.values(),
        labels=data.keys(),
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
    )

    # Amélioration du style du camembert
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=12)
    plt.title(f"{title} class distribution", pad=20)
    plt.axis("equal")

    # Graphique en barres (à droite)
    plt.subplot(1, 2, 2)
    bars = plt.bar(list(data.keys()), list(data.values()), color=colors)

    plt.xlabel("", fontsize=12, labelpad=10)
    plt.ylabel("Images", fontsize=12, labelpad=10)

    # Rotation et ajustement des labels
    plt.xticks(rotation=45, ha="right")

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Grille en arrière-plan pour le graphique en barres
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Ajuster la mise en page
    plt.tight_layout()

    # Créer le dossier output s'il n'existe pas
    dir_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(dir_path, "output")
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder et afficher la figure complète
    filename = f"distribution_combined_{title.lower()}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.show(block=True)  # Attendre que l'utilisateur ferme la fenêtre
    plt.close()


def main() -> None:
    """
    Main function to process the dataset and generate visualizations.
    """
    if len(sys.argv) != 2:
        print("Usage: ./Distribution.py <directory>")
        print("Example: ./Distribution.py ./input/Apple")
        sys.exit(1)

    try:
        directory = sys.argv[1]
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory '{directory}' does not exist")

        dataset_name = path.name
        print(f"\nAnalyzing dataset '{dataset_name}'...")

        # Get statistics using TensorFlow
        image_counts = get_dataset_info(directory)

        # Generate visualizations
        print("\nGenerating charts...")
        try:
            create_charts(image_counts, dataset_name)
        except Exception as e:
            raise RuntimeError(f"Failed to create charts: {str(e)}")

        # Display statistics
        total_images = sum(image_counts.values())
        if total_images == 0:
            raise ValueError("No images found in the dataset")

        print(f"\nStatistics for {dataset_name}:")
        print(f"Total images: {total_images:,}")

        # Calculate and display class distribution
        for category, count in sorted(image_counts.items()):
            percentage = (count / total_images) * 100
            print(f"{category}: {count:,} images ({percentage:.1f}%)")

        # Check for class imbalance
        mini = min(image_counts.values())
        maxi = max(image_counts.values())
        imbalance_ratio = maxi / mini if mini > 0 else float("inf")

        if imbalance_ratio > 3:
            print(
                f"\nWarning: Significant class imbalance detected "
                f"(ratio {imbalance_ratio:.1f}:1)"
            )
            print("Consider using data augmentation to balance classes")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except ValueError as e:
        print(f"\nValidation error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
