# 🌿 Leaffliction
## Description

Leaffliction est un projet de vision par ordinateur qui permet de :
- Analyser la distribution des images dans un dataset
- Équilibrer les classes via des augmentations d'images
- Détecter et classifier les maladies des feuilles

## Structure du projet

```
Leaffliction/
├── src/                          # Scripts Python
│   ├── Augmentation.py          # Augmentation d'images
│   ├── Transformation.py        # Analyse morphologique PlantCV
│   └── Distribution.py          # Analyse de distribution
├── input/                        # Données d'entrée
│   └── Apple/                   # Dataset des feuilles de pommier
│       ├── Apple_Black_rot/
│       ├── Apple_healthy/
│       ├── Apple_rust/
│       └── Apple_scab/
├── output/                       # Résultats générés
│   ├── augmented_directory/     # Images augmentées (traitement par lot)
│   ├── all_transformations.png  # Visualisation des transformations
│   └── distribution_*.png       # Graphiques de distribution
├── requirements.txt             # Dépendances Python
└── README.md                    # Documentation
```

## Installation

### Prérequis
Ce projet nécessite **Python 3.11** pour la compatibilité avec PlantCV et les dépendances scientifiques.

#### Installation de Python 3.11 (macOS)
```bash
# Installer Python 3.11 via Homebrew
brew install python@3.11
```

### Configuration de l'environnement

1. Créez un environnement virtuel Python 3.11 :
```bash
# Utiliser explicitement Python 3.11
python3.11 -m venv .venv_py311
source .venv_py311/bin/activate
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

**Note :** Si vous avez déjà un environnement `.venv` avec une autre version de Python, supprimez-le et recréez-le avec Python 3.11 pour éviter les problèmes de compatibilité.

## Utilisation des scripts

### Distribution.py
Analyse la distribution des images dans les classes.

**Emplacement :** `/src/Distribution.py`

```bash
python src/Distribution.py <chemin_dossier>
# Exemple : python src/Distribution.py ./input/Apple
```

**Sortie :**
- Affiche la distribution des classes dans le terminal
- Génère des graphiques (camembert et histogramme)
- Sauvegarde les visualisations dans `output/`

### Augmentation.py
Équilibre les classes en générant des images transformées avec 6 types d'augmentation.

**Emplacement :** `/src/Augmentation.py`

```bash
# Pour une image unique (depuis le dossier src/) :
python src/Augmentation.py <chemin_image>

# Pour un dossier avec nombre cible d'images par classe :
python src/Augmentation.py <chemin_dossier> --target <nombre>

# Exemples :
python src/Augmentation.py ./input/Apple/apple_healthy/image.jpg
python src/Augmentation.py ./input/Apple --target 1640
```

**Sortie :** Les images augmentées sont sauvegardées dans `output/` (image unique) ou `output/augmented_directory/` (traitement par lot).

#### Les 6 types d'augmentation (valeurs fixes)

1. **Rotation** : Rotation de 25° pour simuler différentes orientations de capture
2. **Blur** : Flou gaussien (radius=2) pour simuler des photos légèrement floues ou en mouvement
3. **Contrast** : Augmentation du contraste (×1.5) pour simuler différentes conditions d'éclairage/capture
4. **Zoom** : Zoom sur le centre de l'image (70% de la surface) pour varier la distance de prise de vue
5. **Brightness** : Augmentation de la luminosité (×1.3) pour simuler différentes conditions d'illumination
6. **Distortion** : Transformation en perspective pour simuler différents angles de vue (effet 3D)

Ces augmentations permettent d'enrichir le dataset et d'améliorer la robustesse du modèle de classification.

### Transformation.py
Analyse morphologique et extraction de caractéristiques des feuilles avec PlantCV.

**Emplacement :** `/src/Transformation.py`

```bash
python src/Transformation.py
```

**Sortie :** Génère une image combinée dans `output/all_transformations.png` avec 6 transformations.

#### Les 6 Transformations et leur intérêt

##### 1. **Original Image** (Image Originale)
**Intérêt :** Image de référence
- Point de départ pour toutes les analyses
- Permet de comparer visuellement les résultats des transformations
- Montre l'état réel de la feuille avec ses taches/maladies

##### 2. **Gaussian Blur** (Flou Gaussien)
**Intérêt :** Réduction du bruit et prétraitement
- Lisse les petites imperfections et le bruit de l'image
- Améliore la segmentation en rendant les transitions de couleur plus douces
- Réduit les faux positifs lors de la détection de contours
- Utile avant la création du masque pour éviter les petits trous

##### 3. **Binary Mask** (Masque Binaire)
**Intérêt :** Segmentation fond/objet
- Sépare la feuille du fond (blanc = feuille, noir = fond)
- Base essentielle pour toutes les analyses suivantes
- Permet de mesurer uniquement la feuille (pas le fond)
- Identifie automatiquement la région d'intérêt
- Important pour calculer la surface réelle de la feuille

##### 4. **ROI Objects** (Région d'Intérêt)
**Intérêt :** Isolation de l'objet à analyser
- Extrait uniquement la feuille en supprimant complètement le fond
- Facilite la visualisation et l'analyse de la feuille seule
- Préparation pour le machine learning : images normalisées sans bruit de fond
- Permet de voir clairement les zones malades sur la feuille
- Utile pour comparer plusieurs feuilles sans interférence du fond

##### 5. **Object Analysis** (Analyse d'Objet)
**Intérêt :** Quantification des propriétés morphologiques
- **Mesures quantitatives** :
  - **Area (Surface)** : Taille de la feuille - utile pour détecter le flétrissement
  - **Perimeter (Périmètre)** : Longueur du contour - détecte les bords irréguliers/mangés
  - **Circularity (Circularité)** : Forme régulière ou déformée - indicateur de santé
- **Contour vert** : Délimitation exacte de la feuille
- **Rectangle bleu** : Boîte englobante pour dimensionnement
- **Ellipse jaune** : Forme idéale pour comparaison
- Ces métriques peuvent détecter des anomalies (ex: feuille trop petite = maladie)

##### 6. **Pseudolandmarks** (Points Caractéristiques)
**Intérêt :** Analyse de la forme et des déformations
- Points équidistants le long du contour pour analyser la forme
- **Centre (cyan)** : Point de référence pour les mesures
- **Points rouges/bleus** : Marquent des positions spécifiques
- Utile pour :
  - Détecter les déformations de la feuille (comparaison avec feuille saine)
  - Analyse statistique de la forme (symétrie, régularité)
  - Machine learning : features pour classifier les maladies
  - Suivi temporel : évolution de la forme dans le temps

#### Pipeline d'analyse
```
Gaussian Blur → Nettoie l'image
Binary Mask → Isole la feuille
ROI Objects → Prépare pour l'analyse
Object Analysis → Mesure les symptômes (taille, forme, déformation)
Pseudolandmarks → Analyse fine de la géométrie
```

Ces transformations permettent de **quantifier objectivement** l'état de santé d'une feuille plutôt que de se fier à l'œil humain, ce qui est essentiel pour un système de détection automatique de maladies !


## Workflow recommandé

1. **Analyser la distribution** :
   ```bash
   python src/Distribution.py ./input/Apple
   ```

2. **Équilibrer les classes** :
   ```bash
   python src/Augmentation.py ./input/Apple --target 1640
   ```

3. **Analyser les caractéristiques** :
   ```bash
   python src/Transformation.py
   ```

4. **Nettoyer si nécessaire** :
   ```bash
   python clean_augmented.py ./output/augmented_directory/Apple
   ```

