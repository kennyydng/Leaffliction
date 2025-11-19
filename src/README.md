# 📚 Notice d'utilisation des scripts

Ce document détaille l'utilisation de chaque script du projet Leaffliction.

## Table des matières

- [Distribution.py](#distributionpy) - Analyse de distribution
- [Augmentation.py](#augmentationpy) - Augmentation d'images
- [Transformation.py](#transformationpy) - Analyse morphologique
- [Scripts utilitaires](#scripts-utilitaires)

---

## Distribution.py

### Description
Analyse la distribution des images dans un dataset et génère des visualisations graphiques.

### Usage

```bash
python src/Distribution.py <chemin_dossier>
```

### Exemples

```bash
# Analyser le dataset Apple complet
python src/Distribution.py ./input/Apple

# Analyser un autre dataset
python src/Distribution.py ./input/Grape
```

### Entrées
- **Dossier** : Chemin vers un répertoire contenant des sous-dossiers (classes)
- **Structure attendue** :
  ```
  input/Apple/
  ├── Apple_Black_rot/
  │   ├── image1.jpg
  │   ├── image2.jpg
  │   └── ...
  ├── Apple_healthy/
  ├── Apple_rust/
  └── Apple_scab/
  ```

### Sorties
- **Terminal** : Statistiques détaillées avec comptage et pourcentages
- **Fichier** : `output/distribution_combined_<dataset>.png`
  - Graphique en camembert (répartition en %)
  - Graphique en barres (nombre d'images)

### Informations affichées
- Nombre total d'images
- Nombre d'images par classe
- Pourcentage de chaque classe
- ⚠️ Avertissement si déséquilibre détecté (ratio > 3:1)

### Exemple de sortie

```
Analyzing dataset 'Apple'...

Generating charts...

Statistics for Apple:
Total images: 3,164
Apple_Black_rot: 620 images (19.6%)
Apple_healthy: 1,640 images (51.8%)
Apple_rust: 275 images (8.7%)
Apple_scab: 629 images (19.9%)

Warning: Significant class imbalance detected (ratio 6.0:1)
Consider using data augmentation to balance classes
```

---

## Augmentation.py

### Description
Génère des images augmentées pour équilibrer les classes d'un dataset ou traiter une image unique.

### Usage

```bash
# Mode 1 : Image unique
python src/Augmentation.py <chemin_image>

# Mode 2 : Dataset complet avec target
python src/Augmentation.py <chemin_dossier> --target <nombre>
```

### Exemples

```bash
# Augmenter une seule image (génère 6 variantes)
python src/Augmentation.py ./input/Apple/Apple_Black_rot/image1.jpg

# Équilibrer toutes les classes à 1640 images
python src/Augmentation.py ./input/Apple --target 1640

# Équilibrer à 2000 images par classe
python src/Augmentation.py ./input/Apple --target 2000
```

### Entrées

#### Mode image unique
- **Image** : Chemin vers un fichier image (JPG, JPEG, PNG)

#### Mode dataset
- **Dossier** : Chemin vers un répertoire contenant des classes
- **--target** : Nombre cible d'images par classe (optionnel)

### Sorties

#### Mode image unique
- **Dossier** : `output/`
- **Fichiers générés** : 6 images augmentées
  - `image_rotation.jpg`
  - `image_blur.jpg`
  - `image_contrast.jpg`
  - `image_zoom.jpg`
  - `image_brightness.jpg`
  - `image_distortion.jpg`

#### Mode dataset
- **Dossier** : `output/augmented_directory/<nom_dataset>/`
- **Structure** : Même hiérarchie que l'entrée avec images originales + augmentées

### Les 6 types d'augmentation

| Augmentation | Description | Paramètres |
|-------------|-------------|------------|
| **Rotation** | Rotation de l'image | 25° avec fond gris |
| **Blur** | Flou gaussien | Radius = 2 |
| **Contrast** | Augmentation du contraste | Facteur ×1.5 |
| **Zoom** | Zoom sur le centre | Crop 80% de l'image centrale |
| **Brightness** | Augmentation de luminosité | Facteur ×1.3 |
| **Distortion** | Transformation perspective | Coefficients de distorsion réduits |

### Fonctionnement du mode dataset

1. **Analyse** : Compte les images dans chaque classe
2. **Copie** : Copie toutes les images originales
3. **Génération** : Pour chaque classe sous le target :
   - Calcule le nombre d'images à générer
   - Applique les 6 augmentations cycliquement
   - Continue jusqu'à atteindre le target

### Exemple de sortie

```bash
$ python src/Augmentation.py ./input/Apple --target 1640

Apple_Black_rot: Generating 1020 additional images
Saved output/augmented_directory/Apple/Apple_Black_rot/image1_rotation.JPG
Saved output/augmented_directory/Apple/Apple_Black_rot/image1_blur.JPG
...
Apple_rust: Generating 1365 additional images
...
Done: 2385 new images generated
```

---

## Transformation.py

### Description
Effectue une analyse morphologique complète d'une feuille avec PlantCV et génère une visualisation des 6 transformations.

### Usage

```bash
python src/Transformation.py
```

### Entrées
- **Image par défaut** : `input/Apple/Apple_Black_rot/image (1).JPG`
- Pour modifier l'image analysée, éditez la variable `image_path` dans le script

### Sorties
- **Fichier** : `output/all_transformations.png`
- **Contenu** : Grille 2×3 avec 6 transformations annotées

### Les 6 transformations

| # | Transformation | Description | Utilité |
|---|---------------|-------------|---------|
| 1 | **Original Image** | Image brute | Référence visuelle |
| 2 | **Gaussian Blur** | Flou gaussien | Réduction du bruit avant segmentation |
| 3 | **Binary Mask** | Masque binaire | Séparation feuille/fond (blanc/noir) |
| 4 | **ROI Objects** | Région d'intérêt | Isolation de la feuille |
| 5 | **Object Analysis** | Analyse d'objet | Mesures : surface, périmètre, circularité |
| 6 | **Pseudolandmarks** | Points caractéristiques | Analyse de la forme et déformations |

### Pipeline d'analyse

```
Image Originale
    ↓
Gaussian Blur (réduction bruit)
    ↓
Binary Mask (segmentation)
    ↓
ROI Objects (isolation)
    ↓
Object Analysis (mesures quantitatives)
    ↓
Pseudolandmarks (analyse géométrique)
```

### Métriques extraites

L'étape **Object Analysis** fournit :
- **Area** : Surface de la feuille (pixels²)
- **Perimeter** : Périmètre du contour (pixels)
- **Circularity** : Indice de forme (0-1)
- **Bounding box** : Rectangle englobant
- **Ellipse** : Forme elliptique ajustée

### Exemple de sortie

```
Image loaded: /Users/.../input/Apple/Apple_Black_rot/image (1).JPG
Combined image saved: /Users/.../output/all_transformations.png
```

---

## Scripts utilitaires

### clean_augmented.py

Nettoie les images augmentées d'un répertoire.

```bash
python clean_augmented.py <chemin_dossier>
```

**Exemple** :
```bash
# Nettoyer un dossier spécifique
python clean_augmented.py ./input/Apple/Apple_Black_rot

# Nettoyer les images générées
python clean_augmented.py ./output/augmented_directory/Apple
```

**Action** : Supprime tous les fichiers contenant les suffixes :
- `_rotation`
- `_blur`
- `_contrast`
- `_zoom`
- `_brightness`
- `_distortion`

### clean_whitespace.py

Remplace les espaces par des underscores dans les noms de fichiers.

```bash
python clean_whitespace.py <chemin_dossier>
```

**Exemple** :
```bash
python clean_whitespace.py ./input/Apple
```

**Action** :
- `image (1).jpg` → `image_(1).jpg`
- `my file.png` → `my_file.png`

---

## Workflow complet

### 1️⃣ Analyser votre dataset

```bash
python src/Distribution.py ./input/Apple
```

→ Observez les déséquilibres de classes

### 2️⃣ Équilibrer les classes

```bash
python src/Augmentation.py ./input/Apple --target 1640
```

→ Génère des images jusqu'à atteindre 1640 par classe

### 3️⃣ Analyser les caractéristiques

```bash
python src/Transformation.py
```

→ Génère une visualisation des transformations morphologiques

### 4️⃣ (Optionnel) Nettoyer

```bash
# Nettoyer les images augmentées si nécessaire
python clean_augmented.py ./output/augmented_directory/Apple
```

---

## Dépendances requises

Tous les scripts nécessitent :
- Python 3.11
- matplotlib
- numpy
- Pillow (PIL)
- opencv-python
- plantcv (pour Transformation.py uniquement)

Installation :
```bash
pip install -r requirements.txt
```

---

## Troubleshooting

### Problème : "No module named 'plantcv'"
**Solution** :
```bash
source .venv_py311/bin/activate
pip install plantcv
```

### Problème : "Cannot load image"
**Vérifications** :
- Le chemin de l'image est correct
- L'extension est supportée (.jpg, .jpeg, .png)
- Les permissions de lecture sont correctes

### Problème : "No valid images found"
**Vérifications** :
- Le dossier contient des sous-dossiers (classes)
- Les sous-dossiers contiennent des images
- Les extensions sont correctes

### Problème : Erreurs d'importation matplotlib/numpy
**Solution** : Vérifier que vous utilisez Python 3.11
```bash
python --version  # Doit afficher 3.11.x
```

---

## Support

Pour plus d'informations, consultez le [README principal](../README.md) ou ouvrez une issue sur GitHub.
