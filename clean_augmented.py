#!/usr/bin/env python3
import os
from pathlib import Path
import argparse


def clean_augmented_images(directory_path):
    """
    Supprime toutes les images augmentées
    dans le répertoire et ses sous-dossiers.
    Les images augmentées sont identifiées
    par les suffixes _flip, _rotate, etc.
    """
    directory = Path(directory_path)
    if not directory.exists():
        print(f"Erreur : Le répertoire {directory} n'existe pas")
        return

    # Suffixes des images augmentées
    suffixes = ["_flip",
                "_rotate",
                "_skew",
                "_shear",
                "_crop",
                "_distort"]

    # Compteur des fichiers supprimés
    deleted_count = 0

    # Extensions d'images à rechercher
    image_extensions = [".jpg", ".JPG", ".jpeg", ".JPEG"]

    print(f"Nettoyage du répertoire : {directory}")

    # Parcourir tous les fichiers
    for ext in image_extensions:
        for img_path in directory.rglob(f"*{ext}"):
            # Vérifier si le fichier est une image augmentée
            if any(suffix in img_path.stem for suffix in suffixes):
                try:
                    os.remove(img_path)
                    deleted_count += 1
                    if (
                        deleted_count % 100 == 0
                    ):  # Afficher le progrès tous les 100 fichiers
                        print(f"Fichiers supprimés : {deleted_count}")
                except Exception as e:
                    print(f"Erreur lors de la suppression de {img_path}: {e}")

    print(f"\nNettoyage terminé : {deleted_count} fichiers supprimés")


def main():
    parser = argparse.ArgumentParser(
        description="Nettoie les images augmentées d'un répertoire"
    )
    parser.add_argument("directory", help="Path to clean")
    args = parser.parse_args()

    clean_augmented_images(args.directory)


if __name__ == "__main__":
    main()
