"""
train_model.py
----------------
This script trains a CNN model to classify eye images as
"Awake" or "Sleepy". The trained model is saved as models/eye_cnn_model.h5
and is later used by detect.py for real-time prediction.

Dataset used: MRL Infrared Eye Images Dataset (Awake / Sleepy)
Folder structure (already provided by the dataset):
archive/data/train/awake
archive/data/train/sleepy
archive/data/val/awake
archive/data/val/sleepy
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ----------------------------
# Configuration
# ----------------------------
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
BATCH_SIZE = 32
NUM_EPOCHS = 3  # start small to test the pipeline; increase later (e.g. 10-15) for the final model

TRAIN_DIR = r"C:\Users\Dell\Downloads\archive\data\train"
VALID_DIR = r"C:\Users\Dell\Downloads\archive\data\val"
MODEL_SAVE_PATH = "models/eye_cnn_model.h5"


def build_cnn_model():
    model = Sequential()

    # First convolution block
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second convolution block
    model.add(Conv2D(32, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third convolution block
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation="sigmoid"))  # binary output: awake vs sleepy

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def get_data_generators():
    # grayscale, rescaled pixel values between 0 and 1
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.2,
        shear_range=0.2,
        horizontal_flip=True
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    valid_generator = valid_datagen.flow_from_directory(
        VALID_DIR,
        target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    return train_generator, valid_generator


def main():
    if not os.path.exists("models"):
        os.makedirs("models")

    train_generator, valid_generator = get_data_generators()

    # class_indices tells us which folder got mapped to 0 and which to 1
    # e.g. {'awake': 0, 'sleepy': 1}
    print("Class label mapping:", train_generator.class_indices)

    model = build_cnn_model()
    model.summary()

    model.fit(
        train_generator,
        epochs=NUM_EPOCHS,
        validation_data=valid_generator
    )

    model.save(MODEL_SAVE_PATH)
    print("Model saved at:", MODEL_SAVE_PATH)


if __name__ == "__main__":
    main() 