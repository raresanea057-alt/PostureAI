# 🧍 PostureAI

<div align="center">

### Real-Time Posture Detection with Computer Vision

Detect poor posture from your webcam using **MediaPipe**, **OpenCV**, and a trained machine learning model.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-FF6F00?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Demo

<p align="center">
  <img src="assets/demo.gif" width="850" alt="Demo">
</p>

---

## Features

- Real-time posture detection from a webcam
- 5-second personal calibration
- Smoothed predictions to reduce flickering
- Detects slouching and shoulder imbalance
- Runs entirely offline

---

## Tech Stack

- Python
- MediaPipe
- OpenCV
- NumPy
- scikit-learn
- Joblib

---

## Installation

Clone the repository:

```bash
git clone https://github.com/raresanea057-alt/PostureAI.git
cd PostureAI
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python posture_check.py
```

The MediaPipe model is downloaded automatically the first time you run the project.

---

## How it works

1. Capture webcam frames.
2. Extract 33 body landmarks with MediaPipe.
3. Compute posture-related geometric features.
4. Normalize the features.
5. Predict posture using the trained model.
6. Smooth predictions over recent frames.
7. Display the result in real time.

---

## Project Structure

```text
PostureAI/
├── posture_check.py
├── model/
│   ├── posture_model_optuna.pkl
│   ├── scaler_optuna.pkl
│   └── pose_landmarker_lite.task
├── mediapipe.ipynb
├── model.ipynb
├── requirements.txt
└── README.md
```

---

## Model

- Validation F1 score: **0.9229**
- Trained on the **Multi-Gait & Posture** dataset
- Hyperparameters optimized with Optuna

---

## License

MIT
