# 🌿 Leaffliction

## À propos du projet

Leaffliction est un système de vision par ordinateur dédié à l'analyse et à la classification des maladies foliaires, en particulier sur les feuilles de plantes. Ce projet combine des techniques de traitement d'images, d'analyse morphologique et d'augmentation de données pour créer un pipeline complet de préparation de dataset.

### Fonctionnalités principales

#### 1. Analyse de distribution
- Comptage automatique des images par classe
- Visualisation de la répartition (camembert et histogramme)
- Détection de déséquilibres dans le dataset

#### 2. Augmentation d'images
- 6 types d'augmentation : rotation, blur, contrast, zoom, brightness, distortion
- Équilibrage automatique des classes
- Valeurs d'augmentation optimisées pour préserver le réalisme

#### 3. Analyse morphologique
- Extraction de caractéristiques avec PlantCV
- 6 transformations : gaussian blur, masque binaire, ROI, analyse d'objet, pseudolandmarks
- Quantification objective de l'état de santé des feuilles

### Dataset utilisé

Le projet utilise le dataset **Plant Village - Apple Leaf Disease** qui contient 4 classes :
- **Apple_healthy** : Feuilles saines (51.8% du dataset)
- **Apple_Black_rot** : Pourriture noire (19.6%)
- **Apple_scab** : Tavelure (19.9%)
- **Apple_rust** : Rouille (8.7%)

**Total** : 3,164 images

### Technologies utilisées

- **Python 3.11** : Langage principal
- **OpenCV** : Traitement d'images
- **PlantCV** : Analyse morphologique spécialisée pour les plantes
- **Matplotlib** : Visualisation de données
- **Pillow (PIL)** : Manipulation d'images
- **NumPy** : Calculs numériques

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

## Démarrage rapide

```bash
# 1. Cloner le projet
git clone https://github.com/kennyydng/Leaffliction.git
cd Leaffliction

# 2. Installer Python 3.11 (si nécessaire)
brew install python@3.11

# 3. Créer l'environnement virtuel
python3.11 -m venv .venv_py311
source .venv_py311/bin/activate

# 4. Installer les dépendances
pip install -r requirements.txt

# 5. Analyser la distribution du dataset
python src/Distribution.py ./input/Apple

# 6. Équilibrer les classes (optionnel)
python src/Augmentation.py ./input/Apple --target 1640

# 7. Analyser les caractéristiques morphologiques
python src/Transformation.py
```

## Utilisation

Pour des instructions détaillées sur l'utilisation de chaque script, consultez le [README dans src/](src/README.md).

### Aperçu des scripts

- **Distribution.py** : Analyse et visualise la distribution des classes
- **Augmentation.py** : Génère des images augmentées pour équilibrer le dataset
- **Transformation.py** : Effectue une analyse morphologique avec PlantCV

## Résultats

Tous les résultats sont sauvegardés dans le dossier `output/` :
- Graphiques de distribution
- Images augmentées
- Visualisations des transformations morphologiques

## Workflow recommandé

1. **Analyser** : Utilisez `Distribution.py` pour comprendre votre dataset
2. **Équilibrer** : Si nécessaire, utilisez `Augmentation.py` pour équilibrer les classes
3. **Analyser** : Utilisez `Transformation.py` pour extraire des caractéristiques

## Contributions

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou une pull request.

## Licence

Ce projet est sous licence MIT.

## Auteur

Kenny Duong - [@kennyydng](https://github.com/kennyydng)

## Remerciements

- Dataset : Plant Village Apple Leaf Disease
- PlantCV pour les outils d'analyse morphologique
- OpenCV pour le traitement d'images

