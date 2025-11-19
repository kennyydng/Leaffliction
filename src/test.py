import os
import numpy as np
from tensorflow import keras
import json
from Transformation import save_mask_only

# Set your constants
IMAGE_SIZE = 128  # or whatever your model expects
ROOT_DIR = "./input/valid"  # directory to walk through
DIR = "./output/training/dataset"  # directory to save processed images

save_mask_only(ROOT_DIR, DIR)

with open("output/training/class_names.json", "r") as f:
    CLASS_NAMES = json.load(f)

# Tracking
total = 0
correct = 0

model2 = keras.models.load_model("./output/training/my_model.keras")

# Go through every image in the directory tree
for root, dirs, files in os.walk(DIR):
    for file in files:
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):  # check for valid image formats
            img_path = os.path.join(root, file)
            true_class = os.path.basename(root)

            # Load and preprocess image
            try:
                img = keras.utils.load_img(
                    img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE)
                )
                img_array = keras.utils.img_to_array(img) / 255.0
                img_array = np.expand_dims(
                    img_array, axis=0
                )

                # Predict
                pred = model2.predict(img_array, verbose=0)
                predicted_class_idx = np.argmax(pred[0])
                predicted_class = CLASS_NAMES[predicted_class_idx]

                total += 1
                if predicted_class == true_class:
                    correct += 1

                accuracy = (correct / total) * 100
                print(f"🎯 Accuracy: {accuracy:.2f}% out of {total} images.")

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
