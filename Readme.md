# Facial Expression Recognition

A CNN-based computer vision system for classifying seven facial expressions from FER-2013 and performing image and real-time webcam inference.

## Features

- FER-2013 facial-expression classification
- Grayscale 48×48 input preprocessing
- Training-time data augmentation
- CNN with Batch Normalization and Dropout
- Early stopping and adaptive learning rate
- Best-model checkpointing
- Classification report and confusion matrix
- Standalone image prediction
- Real-time webcam face detection and emotion prediction

## Emotions

Angry · Disgust · Fear · Happy · Sad · Surprise · Neutral

## Architecture

`Input → Conv/BN/Pool × 4 → Dense → Softmax`

The model uses four convolutional blocks with increasing feature depth (32, 64, 128, 256 filters), followed by a 256-unit dense layer and a seven-class softmax output.

## Project Structure

```text
Facial Expression Recognition/
├── facial_expression_recognition.py   # Original internship implementation
├── model.py                            # Modular CNN definition
├── train.py                            # Training + evaluation pipeline
├── predict.py                          # Single-image inference
├── webcam.py                           # Real-time webcam inference
├── requirements.txt
├── models/                             # Generated trained model
├── results/                            # Generated metrics and plots
└── Facial Expression Recognition.pdf   # Internship report
```

## Dataset

The project uses FER-2013, with 48×48 grayscale facial images across seven emotion classes. The dataset is not included in this repository. Place the extracted dataset at:

```text
fer2013/
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── neutral/
│   ├── sad/
│   └── surprise/
└── test/
    └── ...
```

## Setup

```bash
cd "Facial Expression Recognition"
pip install -r requirements.txt
```

## Train and Evaluate

```bash
python train.py
```

The training pipeline saves the best model to `models/facial_expression_model.keras` and evaluation artifacts to `results/`.

## Predict an Image

```bash
python predict.py path/to/face.jpg
```

## Run Webcam Recognition

First train the model, then:

```bash
python webcam.py
```

Press **Q** to exit.

## Evaluation

The pipeline reports validation accuracy and produces:

- `results/classification_report.txt`
- `results/confusion_matrix.png`
- `results/training_history.png`
- `results/metrics.json`

Accuracy should be reported from the generated `metrics.json` for the exact training run rather than using an unverified estimate.

## Technologies

Python · TensorFlow/Keras · OpenCV · NumPy · scikit-learn · Matplotlib

## Original Internship Work

The original `facial_expression_recognition.py` remains in the project as the baseline implementation. The newer modular files provide a cleaner, reproducible training and inference workflow while preserving the original project concept.
