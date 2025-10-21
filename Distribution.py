#!/usr/bin/env python3
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# Configure TensorFlow logging before import
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Matplotlib configuration
plt.rcParams.update({
    'figure.autolayout': True,
    'axes.labelweight': 'bold',
    'axes.labelsize': 'large',
    'axes.titleweight': 'bold',
    'axes.titlesize': 18,
    'axes.titlepad': 10
})
plt.ion()  # Enable interactive mode

# Constants
IMAGE_SIZE = [128, 128]  # Standard size for all images


def get_dataset_info(directory: str) -> dict:
    """
    Analyze image distribution in the given directory.

    Args:
        directory (str): Path to the dataset directory

    Returns:
        dict: Category names as keys and image counts as values

    Raises:
        SystemExit: If directory cannot be processed
    """
    try:
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            image_size=IMAGE_SIZE,
            shuffle=False,
            batch_size=None
        )

        class_names = dataset.class_names
        image_counts = {}

        # Count images per class
        for images, labels in dataset:
            class_idx = np.argmax(labels.numpy())
            class_name = class_names[class_idx]
            image_counts[class_name] = image_counts.get(class_name, 0) + 1

        if not image_counts:
            raise ValueError("No images found in the directory")

        return image_counts

    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        sys.exit(1)


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
        autopct='%1.1f%%',
        startangle=90,
        colors=colors
    )

    # Amélioration du style du camembert
    plt.setp(autotexts, size=10, weight="bold")
    plt.setp(texts, size=12)
    plt.title(f'{title} class distribution', pad=20)
    plt.axis('equal')

    # Graphique en barres (à droite)
    plt.subplot(1, 2, 2)
    bars = plt.bar(
        list(data.keys()),
        list(data.values()),
        color=colors
    )

    plt.xlabel('', fontsize=12, labelpad=10)
    plt.ylabel('Images', fontsize=12, labelpad=10)

    # Rotation et ajustement des labels
    plt.xticks(rotation=45, ha='right')

    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Grille en arrière-plan pour le graphique en barres
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    # Ajuster la mise en page
    plt.tight_layout()

    # Créer le dossier output s'il n'existe pas
    dir_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(dir_path, 'output')
    os.makedirs(output_dir, exist_ok=True)

    # Sauvegarder et afficher la figure complète
    filename = f'distribution_combined_{title.lower()}.png'
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show(block=True)  # Attendre que l'utilisateur ferme la fenêtre
    plt.close()


def main() -> None:
    """
    Main function to process the dataset and generate visualizations.
    """
    if len(sys.argv) != 2:
        print("Usage: ./Distribution.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    if not Path(directory).is_dir():
        print(f"Error: '{directory}' is not a valid directory.")
        sys.exit(1)

    try:
        # Get dataset name from directory
        dataset_name = Path(directory).name
        print(f"\nAnalyzing dataset '{dataset_name}'...")

        # Get statistics using TensorFlow
        image_counts = get_dataset_info(directory)

        # Generate visualizations
        print("\nGenerating charts...")
        create_charts(image_counts, dataset_name)

        # Display statistics
        total_images = sum(image_counts.values())
        print(f"\nStatistics for {dataset_name}:")
        print(f"Total images: {total_images:,}")

        for category, count in sorted(image_counts.items()):
            percentage = (count / total_images) * 100
            print(f"{category}: {count:,} images ({percentage:.1f}%)")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
