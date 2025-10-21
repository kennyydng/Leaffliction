#!/usr/bin/env python3
import os
import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pathlib import Path

# Configuration matplotlib pour l'affichage interactif
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
warnings.filterwarnings("ignore")
plt.ion()  # Active le mode interactif pour l'affichage

# Constants
IMAGE_SIZE = [128, 128]
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}

def get_dataset_info(directory):
    """
    Utilise TensorFlow pour charger et analyser le dataset.
    Retourne un dictionnaire avec les statistiques par classe.
    """
    try:
        dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels='inferred',
            label_mode='categorical',
            image_size=IMAGE_SIZE,
            shuffle=False,
            batch_size=None  # Pour compter toutes les images
        )
        
        class_names = dataset.class_names
        image_counts = {}
        
        # Compter les images par classe
        for images, labels in dataset:
            class_idx = np.argmax(labels.numpy())
            class_name = class_names[class_idx]
            image_counts[class_name] = image_counts.get(class_name, 0) + 1
            
        return image_counts
        
    except Exception as e:
        print(f"Erreur lors du chargement des données : {str(e)}")
        sys.exit(1)
    
    # Parcourir les sous-répertoires directs
    for subdir in directory.iterdir():
        if subdir.is_dir():
            # Nettoyer le nom du sous-répertoire
            category_name = subdir.name.replace(directory.name.lower() + '_', '')
            category_name = category_name.replace('_', ' ').title()
            
            # Compter les images
            count = sum(1 for file in subdir.glob('*')
                       if file.suffix.lower() in image_extensions)
            image_counts[category_name] = count
    
    return image_counts

def create_visualizations(data, title):
    """
    Crée les deux graphiques côte à côte dans une seule figure.
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
    plt.title(f'Distribution des Images - {title}', pad=20)
    plt.axis('equal')
    
    # Ajouter une légende pour le camembert
    plt.legend(
        wedges,
        data.keys(),
        title="Catégories",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )
    
    # Graphique en barres (à droite)
    plt.subplot(1, 2, 2)
    bars = plt.bar(
        list(data.keys()),
        list(data.values()),
        color=colors
    )
    
    plt.title(f'Nombre d\'Images par Catégorie - {title}')
    plt.xlabel('Catégories', fontsize=12, labelpad=10)
    plt.ylabel('Nombre d\'Images', fontsize=12, labelpad=10)
    
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
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Sauvegarder et afficher la figure complète
    output_path = os.path.join(output_dir, f'distribution_combined_{title.lower()}.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    plt.show(block=True)  # Attendre que l'utilisateur ferme la fenêtre
    plt.close()

def create_bar_chart(data, title):
    """
    Crée un graphique en barres des données.
    """
    plt.figure(figsize=(12, 6))
    
    categories = list(data.keys())
    values = list(data.values())
    
    bars = plt.bar(categories, values)
    plt.title(f'Nombre d\'Images par Catégorie - {title}')
    plt.xlabel('Catégories')
    plt.ylabel('Nombre d\'Images')
    
    # Rotation des labels pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')
    
    # Ajouter les valeurs au-dessus des barres
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom')
    
    # Ajuster la mise en page
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plt.savefig(f'distribution_bar_{title.lower()}.png')
    plt.close()



def main():
    if len(sys.argv) != 2:
        print("Usage: ./Distribution.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Erreur: '{directory}' n'est pas un répertoire valide.")
        sys.exit(1)
    
    # Obtenir le nom du répertoire principal
    dataset_name = os.path.basename(os.path.normpath(directory))
    
    print(f"\nAnalyse du dataset '{dataset_name}'...")
    
    # Obtenir les statistiques avec TensorFlow
    image_counts = get_dataset_info(directory)
    
    if not image_counts:
        print(f"Aucune image trouvée dans les sous-répertoires de '{directory}'")
        sys.exit(1)
    
    # Créer les visualisations
    print("\nCréation des visualisations...")
    create_visualizations(image_counts, dataset_name)
    
    # Afficher les statistiques
    total_images = sum(image_counts.values())
    print(f"\nStatistiques pour {dataset_name}:")
    print(f"Total des images: {total_images:,}")
    print("\nDistribution par catégorie:")
    for category, count in image_counts.items():
        percentage = (count / total_images) * 100
        print(f"{category}: {count:,} images ({percentage:.1f}%)")
        
    print(f"\nGraphique sauvegardé dans le dossier 'output' sous:")
    print(f"- distribution_combined_{dataset_name.lower()}.png")

if __name__ == "__main__":
    main()
