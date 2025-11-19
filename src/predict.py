import numpy as np
from tensorflow import keras
import argparse
import json
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from Transformation import Transformation, Options, save_mask_only, getlastname

IMAGE_SIZE = 128


def load_image(image_path):
    """Load image and convert to RGB."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot load image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image loaded: {image_path}")
    return img


def display_result(image_path, predicted_class):
    """Display and save all transformations."""
    # Load and process image
    original = load_image(image_path)

    opt = Options(
        image_path,
        debug=None,
        writeimg=False,
    )
    try:
        transformation = Transformation(opt)
        transformation.original()
        transformation.gaussian_blur()
        transformation.masked()
        enhanced = transformation.masked2
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle(f"Class predicted: {predicted_class}", fontsize=16)

    # Display images
    axs[0].imshow(original)
    axs[0].set_title("Original Image")

    axs[1].imshow(enhanced, cmap="gray")
    axs[1].set_title("Transformed")

    # Remove axes
    for ax in axs.flat:
        ax.set_xticks([])
        ax.set_yticks([])

    # Save and show
    plt.tight_layout()

    # Créer le dossier output à la racine du projet (parent de src/)
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Le path du fichier a predire"
        )
    parser.add_argument("file", help="Le path du fichier a predire")
    args = parser.parse_args()
    if args.file:
        img_path = args.file

    save_mask_only(img_path, "./output/pred")

    with open("output/training/class_names.json", "r") as f:
        CLASS_NAMES = json.load(f)

    model2 = keras.models.load_model("output/training/my_model.keras")

    try:
        img = keras.utils.load_img(
                "./output/pred/" + getlastname(img_path) + "_masked.JPG",
                target_size=(IMAGE_SIZE, IMAGE_SIZE)
            )
        img_array = keras.utils.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        pred = model2.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(pred[0])
        predicted_class = CLASS_NAMES[predicted_class_idx]

        display_result(img_path, predicted_class)

    except Exception as e:
        print(f"Error processing {img_path}: {e}")
