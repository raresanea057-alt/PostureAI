# 🧍 PostureAI

<div align="center">

### Real-Time Posture Detection with Computer Vision

Detect slouching, forward head posture, and shoulder imbalance directly from your webcam using **Python**, **MediaPipe**, and a trained classifier.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-FF6F00?logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📸 Demo

<p align="center">
  <img src="assets/demo.gif" width="850" alt="PostureAI Demo">
</p>

*(Add a demo GIF to `assets/demo.gif` to show the app in action.)*

---

## 🚀 About the Project

PostureAI is a real‑time posture monitoring application that uses your webcam to track body landmarks via **MediaPipe Pose**. It extracts geometric features (head tilt, shoulder slope, elbow positions, etc.) and feeds them into a **trained classifier** to instantly tell you if your posture is **Good** or **Bad**.

All processing is done **locally** – no data is sent to the cloud. It’s privacy‑first, fast, and lightweight.

---

## ✨ Features

- 🎯 **Real‑time pose tracking** – MediaPipe provides 33 body landmarks.
- 🧘 **Personal calibration** – Sit in your best posture for 5 seconds to set a baseline.
- 📊 **Smooth predictions** – Uses a moving average over 1.5 seconds to avoid jitter.
- 🖥️ **Live visual feedback** – Landmarks, skeleton, and status text are overlaid on the video feed.
- ⚠️ **Detects multiple posture issues**:
  - Slouching (forward head posture)
  - Shoulder asymmetry / imbalance
  - Poor elbow alignment
- 🔒 **100% local** – No internet connection required after the first run.
- ⚡ **Lightweight** – Runs on most modern laptops with a webcam.

---

## 🛠️ Built With

- **Python** 3.10+
- **OpenCV** – video capture and display
- **MediaPipe** – pose landmark detection
- **NumPy** – numerical operations
- **scikit‑learn** – model loading and scaling
- **Joblib** – model serialisation

---

## 📦 Installation

### Prerequisites

- Python 3.10 or higher
- A working webcam
- Git (optional, for cloning)

### Step‑by‑Step

1. **Clone the repository**

   ```bash
   git clone https://github.com/raresanea057-alt/PostureAI.git
   cd PostureAI
Install the required packages

bash
pip install -r requirements.txt
If you don't have a requirements.txt, create one with the following content:

text
opencv-python
mediapipe
numpy
joblib
scikit-learn
Then run pip install -r requirements.txt.

Verify the trained model files
The repository includes the pre‑trained model and scaler inside the model/ folder:

text
model/
├── posture_model_optuna.pkl   # trained classifier
└── scaler_optuna.pkl          # feature scaler
If these are missing, you can train your own model using the provided notebooks (model.ipynb or mediapipe.ipynb), or download the pre‑trained files from the Releases page.

Run the application

bash
python posture_check.py
On the first run, the script will automatically download the MediaPipe pose landmarker model (pose_landmarker_lite.task, ~10 MB) from Google’s servers.

🧘 How to Use
Calibration – When the app starts, you’ll see a CALIBRATING screen.
Sit upright in your best posture for 5 seconds. The system records your personal baseline.

Real‑time feedback – After calibration, the app classifies your posture every frame.

Green text and landmarks = Good posture

Red text and landmarks = Bad posture

Press Q on your keyboard to quit the application.

🧠 How It Works
Webcam frames are captured continuously.

MediaPipe Pose extracts 33 body landmarks from each frame.

Landmarks are smoothed using exponential moving average to reduce noise.

14 geometric features are computed from the landmarks:

Head tilt angle (nose vs. vertical)

Shoulder slope (angle between shoulders and horizontal)

Normalised nose distance from shoulder midpoint

Left elbow position relative to left shoulder (Z, X, Y, distance)

Right elbow position relative to right shoulder (Z, X, Y, distance)

Shoulder asymmetry (Y‑axis difference)

Head rotation (projected on horizontal plane)

Elbow‑to‑elbow distance (normalised by shoulder width)

The 14 features are scaled using a pre‑fitted StandardScaler.

A trained classifier (Random Forest / XGBoost) predicts the probability of “Good” posture.

Predictions are averaged over the last 45 frames (~1.5 seconds at 30 fps) to produce a smooth score.

If the smoothed probability exceeds a threshold (default 0.5), the posture is classified as Good, otherwise Bad.

Visual feedback (landmarks, skeleton, status text) is drawn on the frame and displayed.

📂 Project Structure
text
PostureAI/
│
├── posture_check.py              # Main application script
├── model/
│   ├── posture_model_optuna.pkl  # Trained classifier
│   ├── scaler_optuna.pkl         # Scaler for feature normalisation
│   └── pose_landmarker_lite.task # MediaPipe model (auto‑downloaded)
├── mediapipe.ipynb               # Jupyter notebook for development/testing
├── model.ipynb                   # Training notebook (if present)
├── requirements.txt              # Python dependencies
├── LICENSE                       # MIT License
└── README.md                     # This file
📊 Model Performance
The classifier was trained on the Multi‑Gait & Posture dataset (PhysioNet), which contains labelled posture data.

Validation F1 Score: 0.9229

Classes: Good (0), Bad (1)

The model was optimised using Optuna for hyperparameter tuning, resulting in robust performance across different body types and lighting conditions.
