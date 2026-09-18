"""Real-time facial expression recognition with OpenCV."""
from pathlib import Path
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model import EMOTIONS

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "facial_expression_model.keras"


def run(model_path=MODEL_PATH, camera_index=0):
    model = load_model(model_path)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
            for x, y, w, h in faces:
                roi = cv2.resize(gray[y:y+h, x:x+w], (48, 48)).astype("float32") / 255.0
                probabilities = model.predict(roi.reshape(1, 48, 48, 1), verbose=0)[0]
                index = int(np.argmax(probabilities))
                label = f"{EMOTIONS[index]} {probabilities[index]:.0%}"
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, label, (x, max(y-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Facial Expression Recognition | Press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
