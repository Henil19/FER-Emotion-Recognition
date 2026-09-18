"""Predict an emotion from a single face image."""
from pathlib import Path
import argparse
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model import EMOTIONS

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "facial_expression_model.keras"


def predict_image(image_path, model_path=DEFAULT_MODEL):
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    model = load_model(model_path)
    face = cv2.resize(image, (48, 48)).astype("float32") / 255.0
    probabilities = model.predict(face.reshape(1, 48, 48, 1), verbose=0)[0]
    index = int(np.argmax(probabilities))
    return EMOTIONS[index], float(probabilities[index])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict facial expression from an image")
    parser.add_argument("image", help="Path to an image containing a face")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to .keras model")
    args = parser.parse_args()
    emotion, confidence = predict_image(args.image, args.model)
    print(f"Prediction: {emotion}")
    print(f"Confidence: {confidence:.2%}")
