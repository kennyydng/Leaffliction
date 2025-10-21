# 🌿 Leaffliction

**Classification d’images — Reconnaissance de maladies sur feuilles**

---

## Table des matières

- [Description](#description)
- [Objectifs](#objectifs)
- [Structure du dépôt](#structure-du-dépôt)
- [Scripts](#scripts)
- [Livraison / signature](#livraison--signature)
- [Contraintes et bonnes pratiques](#contraintes-et-bonnes-pratiques)
- [Outils et dépendances recommandés](#outils-et-dépendances-recommandés)
- [Workflow conseillé](#workflow-conseillé)
- [Conseils pratiques](#conseils-pratiques)

---

## Description
Leaffliction est un projet de **vision par ordinateur** visant à détecter et classifier des maladies sur des feuilles à partir d’images.  
Le pipeline complet couvre : analyse du dataset, augmentation des données, transformations d’images, entraînement d’un modèle et prédiction.

---

## Objectifs

- Comprendre et analyser un dataset d’images.
- Équilibrer les classes via des augmentations.
- Appliquer des transformations pour extraire des caractéristiques.
- Entraîner un modèle de classification (≥ **90%** de précision sur un jeu de validation ≥ 100 images).
- Produire des scripts réutilisables et robustes.

---

## Structure du dépôt

.
├─ README.md
├─ Distribution.[ext]    # Analyse et visualisation de la distribution des classes
├─ Augmentation.[ext]    # Génération d'images augmentées
├─ Transformation.[ext]  # Transformations et extractions d'images
├─ train.[ext]           # Entraînement du modèle + export (zip)
├─ predict.[ext]         # Prédiction sur une image donnée
└─ signature.txt         # SHA1 du zip contenant dataset+modèle (NE PAS COMMIT LE DATASET)

---

## Scripts

> Nomme les fichiers comme tu veux, mais garde une API et une documentation claire.

### 1) Distribution.[extension]

- But : analyser un répertoire contenant des sous-dossiers par classe et produire des diagrammes (diagramme en secteurs / histogramme) par classe.
- Usage :

```bash
./Distribution.py ./Apple
```

### 2) Augmentation.[extension]

- But : équilibrer le dataset en générant au moins 6 augmentations par image : flip, rotate, skew, shear, crop, distortion.
- Sortie : nouvelles images nommées original_nom_Type.JPG placées dans `augmented_directory` (ou dans le dossier d'origine si demandé).
- Usage :

```bash
./Augmentation.py "./Apple/apple_healthy/image (1).JPG"
```

### 3) Transformation.[extension]

- But : appliquer ≥ 6 transformations (ex : blur gaussien, masque, ROI, histogramme, contours, pseudolandmarks) et afficher / sauvegarder les résultats.
- Usage :

Fichier unique :
```bash
./Transformation.py "./Apple/apple_healthy/image (1).JPG"
```

Répertoire source -> répertoire destination (avec option `-mask`) :
```bash
./Transformation.py -src Apple/apple_healthy/ -dst dst_directory -mask
```

### 4) train.[extension]

- But : entraîner un modèle sur les données (avec split Train/Validation), sauvegarder le modèle et packager le tout dans `model_and_data.zip`.
- Contraintes : la validation doit contenir ≥ 100 images et atteindre ≥ 90% de précision.
- Usage :

```bash
./train.py ./Apple
# Résultat attendu : model_and_data.zip
```

### 5) predict.[extension]

- But : charger le modèle entraîné et prédire la maladie pour une image donnée, afficher image originale + transformation + label prévu.
- Usage :

```bash
./predict.py "./Apple/apple_healthy/image (1).JPG"
```

---

## Livraison / signature

Ne commite jamais le dataset dans le dépôt.

Crée un `.zip` contenant : dataset (utilisé pour le training + validation) et le modèle entraîné. Génère le SHA1 du `.zip` et place la valeur dans `signature.txt`.

Linux :

```bash
sha1sum directory.zip > signature.txt
```

macOS :

```bash
shasum directory.zip > signature.txt
```

Si le hash ne correspond pas pendant l'évaluation → note 0.

---

## Contraintes et bonnes pratiques

- Aucune erreur système (segfault, bus error...) : ton code doit être stable.
- Python recommandé, mais tout langage accepté. Si Python : respecte flake8.
- Documente clairement tes scripts et indique les dépendances.
- Sépare proprement train / validation pour éviter le "cheat".
- Fournis des tests de base pour prouver la robustesse (même non notés, ils servent lors de la soutenance).

---

## Outils et dépendances recommandés

- Langage : Python 3.8+ (recommandé)
- Libs utiles : `numpy`, `opencv-python`, `matplotlib`, `scikit-learn`, `tensorflow` ou `torch`, `plantcv`

Exemple d’installation :

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Exécution (exemple)

Pour éviter les erreurs d'import (par ex. "No module named 'matplotlib'"), créez et activez un environnement virtuel puis installez les dépendances :

```bash
# depuis la racine du projet
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python main.py
```

### Note macOS (Apple Silicon / M1/M2)

Sur macOS Apple Silicon, vous pouvez activer l'accélération GPU Metal en utilisant les paquets fournis par Apple : `tensorflow-macos` et `tensorflow-metal`.

- Ces paquets sont souvent publiés pour des versions spécifiques de Python (généralement 3.10/3.11). Si vous êtes sur Python 3.13, il est possible qu'il n'existe pas encore de wheel — dans ce cas, créez un venv avec Python 3.11.
- Exemple :

```bash
# Créez un venv avec Python 3.11 (si installé via Homebrew)
/opt/homebrew/opt/python@3.11/bin/python3 -m venv .venv311
source .venv311/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install tensorflow-macos tensorflow-metal
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Si `tensorflow-macos` n'est pas disponible pour votre version de Python, soit utilisez Python 3.11, soit restez sur la build CPU standard (`tensorflow`) dans votre venv actuel.
```

---

## Workflow conseillé (simple et efficace)

1. `Distribution.py` → évaluer l’équilibre des classes.
2. `Augmentation.py` → générer des images pour équilibrer.
3. `Transformation.py` → construire features / masques / preprocess.
4. `train.py` → entraîner, valider, exporter `model_and_data.zip`.
5. `predict.py` → tester des images individuelles.

---

## Conseils pratiques

- Vérifie la qualité des augmentations : générer pour générer ne suffit pas.
- Évite le leak entre train/validation (mêmes images modifiées dans les deux sets = fail).
- Documente ton pipeline et conserve les seeds RNG pour reproductibilité.
- Si tu vises ≥ 90% sur validation, prépare-toi à justifier chaque choix (augmentation, architecture, métriques).

---