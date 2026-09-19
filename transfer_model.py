"""MobileNetV2 transfer-learning model for FER-2013."""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2


EMOTIONS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
    "Surprise",
]


def build_transfer_model(input_shape=(48, 48, 3), num_classes=7):
    base_model = MobileNetV2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=input_shape),

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.40),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model