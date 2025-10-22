# 🌿 Leaffliction
## Description

Leaffliction est un projet de vision par ordinateur qui permet de :
- Analyser la distribution des images dans un dataset
- Équilibrer les classes via des augmentations d'images
- Détecter et classifier les maladies des feuilles

## Installation

1. Créez un environnement virtuel Python :
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

Note macOS (Apple Silicon) :
Pour l'accélération GPU, utilisez Python 3.11 et installez les paquets spécifiques :
```bash
python -m pip install tensorflow-macos tensorflow-metal
```

## Utilisation des scripts

### Distribution.py
Analyse la distribution des images dans les classes.
```bash
python Distribution.py <chemin_dossier>
# Exemple : python Distribution.py ./input/Apple
```

Résultat :
- Affiche la distribution des classes
- Génère des graphiques (camembert et histogramme)
- Sauvegarde les visualisations dans `output/`

### Augmentation.py
Équilibre les classes en générant des images transformées.
```bash
# Pour une image unique :
python Augmentation.py <chemin_image>

# Pour un dossier avec nombre cible d'images par classe :
python Augmentation.py <chemin_dossier> --target <nombre>

# Exemples :
python Augmentation.py ./input/Apple/apple_healthy/image.jpg
python Augmentation.py ./input/Apple --target 1640
```