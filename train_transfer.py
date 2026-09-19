"""Train the MobileNetV2 transfer-learning model on FER-2013."""

from pathlib import Path
import json

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from transfer_model import build_transfer_model, EMOTIONS


SEED = 42
IMG_SIZE = (96, 96)
BATCH_SIZE = 64
EPOCHS = 40

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "fer2013"

MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)


train_data = train_datagen.flow_from_directory(
    DATA_DIR / "train",
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
)

val_data = val_datagen.flow_from_directory(
    DATA_DIR / "test",
    target_size=IMG_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
)


expected_indices = {
    name.lower(): index
    for index, name in enumerate(EMOTIONS)
}

if (
    train_data.class_indices != expected_indices
    or val_data.class_indices != expected_indices
):
    raise ValueError(
        "Dataset class order does not match transfer_model.py. "
        f"Expected {expected_indices}, "
        f"got train={train_data.class_indices}, "
        f"test={val_data.class_indices}."
    )


(RESULTS_DIR / "transfer_class_indices.json").write_text(
    json.dumps(train_data.class_indices, indent=2),
    encoding="utf-8",
)


model = build_transfer_model(
    input_shape=(96, 96, 3),
    num_classes=len(EMOTIONS),
)


callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    ),
    ModelCheckpoint(
        MODEL_DIR / "mobilenetv2_fer_model.keras",
        monitor="val_accuracy",
        save_best_only=True,
    ),
]


history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# Save training curves.
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Validation")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Validation")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

fig.tight_layout()
fig.savefig(
    RESULTS_DIR / "mobilenetv2_training_history.png",
    dpi=150,
)
plt.close(fig)


# Evaluate without shuffling.
val_data.reset()

probabilities = model.predict(
    val_data,
    verbose=1,
)

y_pred = np.argmax(probabilities, axis=1)
y_true = val_data.classes

labels = [
    name
    for name, _ in sorted(
        val_data.class_indices.items(),
        key=lambda item: item[1],
    )
]


report = classification_report(
    y_true,
    y_pred,
    target_names=labels,
    digits=4,
)

(
    RESULTS_DIR / "mobilenetv2_classification_report.txt"
).write_text(
    report,
    encoding="utf-8",
)


cm = confusion_matrix(y_true, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)

fig, ax = plt.subplots(figsize=(8, 7))

display.plot(
    ax=ax,
    xticks_rotation=45,
    colorbar=False,
)

ax.set_title("FER-2013 MobileNetV2 Confusion Matrix")

fig.tight_layout()

fig.savefig(
    RESULTS_DIR / "mobilenetv2_confusion_matrix.png",
    dpi=150,
)

plt.close(fig)


metrics = {
    "validation_accuracy": float(
        model.evaluate(val_data, verbose=0)[1]
    ),
    "epochs_trained": len(history.history["loss"]),
    "classes": labels,
    "model": "MobileNetV2",
    "pretrained": True,
    "input_shape": [48, 48, 3],
}


(
    RESULTS_DIR / "mobilenetv2_metrics.json"
).write_text(
    json.dumps(metrics, indent=2),
    encoding="utf-8",
)


print(
    "MobileNetV2 training complete. "
    "Artifacts saved to models/ and results/."
)