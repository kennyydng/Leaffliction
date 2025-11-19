# Imports
import os
import warnings
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from keras.preprocessing import image_dataset_from_directory
import argparse
from Transformation import save_mask_only

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Hide all WARNING/ERROR from TF
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Pretend no GPU exists
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

EPOCHS = 20

parser = argparse.ArgumentParser(
    description="Le path du dossier de data d'entrainement"
)
parser.add_argument("dir", help="Le path du dossier de data d'entrainement")

args = parser.parse_args()
if args.dir:
    DIR = args.dir

save_mask_only(DIR, "./output/training/dataset")


# Reproducability
def set_seed(seed=31415):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


set_seed(31415)

# Set Matplotlib defaults
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=18,
    titlepad=10,
)
plt.rc("image", cmap="magma")
warnings.filterwarnings("ignore")

# Load training and validation sets
ds_train_ = image_dataset_from_directory(
    "./output/training/dataset/leaves",
    labels="inferred",
    label_mode="categorical",
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=64,
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=123,
)
ds_valid_ = image_dataset_from_directory(
    "./output/training/dataset/leaves",
    labels="inferred",
    label_mode="categorical",
    image_size=[128, 128],
    interpolation="nearest",
    batch_size=64,
    shuffle=False,
    validation_split=0.2,
    subset="validation",
    seed=123,
)
CLASS_NAMES = ds_train_.class_names
with open("output/training/class_names.json", "w") as f:
    json.dump(CLASS_NAMES, f)


# Data Pipeline
def convert_to_float(image, label):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


AUTOTUNE = tf.data.experimental.AUTOTUNE
ds_train = ds_train_.map(
        convert_to_float
    ).cache().prefetch(buffer_size=AUTOTUNE)
ds_valid = ds_valid_.map(
        convert_to_float
    ).cache().prefetch(buffer_size=AUTOTUNE)

model = keras.Sequential(
    [
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ]
)

model.compile(
    optimizer=keras.optimizers.Adam(epsilon=0.01),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    ds_train,
    validation_data=ds_valid,
    epochs=EPOCHS,
    verbose=1,
)

history_frame = pd.DataFrame(history.history)
history_frame.loc[:, ["loss", "val_loss"]].plot()
history_frame.loc[:, ["accuracy", "val_accuracy"]].plot()

print(history_frame.tail(1))

model.save("./output/training/my_model.keras")

plt.show()
