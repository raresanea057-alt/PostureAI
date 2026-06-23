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

> Replace `assets/demo.gif` with a recording of the application in action.

---

## 🚀 About the Project

PostureAI is a real-time posture monitoring application that uses your webcam to track body landmarks via **MediaPipe Pose**. It extracts geometric features (head tilt, shoulder slope, elbow positions, etc.) and feeds them into a trained classifier to instantly determine whether your posture is **Good** or **Bad**.

All processing is performed **locally**—no data is uploaded to the cloud.

---

## ✨ Features

- 🎯 Real-time pose tracking using **MediaPipe Pose**
- 🧘 Personal 5-second calibration
- 📊 Moving average prediction smoothing
- 🖥️ Live landmark visualization
- ⚠️ Detects:
  - Slouching
  - Forward head posture
  - Shoulder imbalance
  - Poor elbow alignment
- 🔒 100% local processing
- ⚡ Lightweight and fast

---

## 🛠️ Built With

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Joblib

---

# 📦 Installation

## Prerequisites

- Python 3.10+
- Webcam
- Git (optional)

## 1. Clone the repository

```bash
git clone https://github.com/raresanea057-alt/PostureAI.git
cd PostureAI
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, create one containing:

```text
opencv-python
mediapipe
numpy
joblib
scikit-learn
```

Then install it:

```bash
pip install -r requirements.txt
```

## 3. Verify model files

The repository should contain:

```text
model/
├── posture_model_optuna.pkl
└── scaler_optuna.pkl
```

If these files are missing, either:

- Train your own model using `model.ipynb` or `mediapipe.ipynb`
- Download the pretrained files from the project's Releases page

## 4. Run the application

```bash
python posture_check.py
```

On the first run, MediaPipe automatically downloads:

```text
pose_landmarker_lite.task
```

(~10 MB)

---

# 🧘 How to Use

### Calibration

When the application starts, you'll see a **CALIBRATING** screen.

Sit in your best posture for **5 seconds** while the system records your baseline.

### Real-time Detection

After calibration, every frame is classified as:

- 🟢 **Good posture**
- 🔴 **Bad posture**

Press **Q** to quit.

---

# 🧠 How It Works

1. Webcam frames are captured.
2. MediaPipe Pose extracts **33 body landmarks**.
3. Landmarks are smoothed using an exponential moving average.
4. Fourteen geometric posture features are computed.
5. Features are normalized with a pretrained `StandardScaler`.
6. The classifier predicts posture quality.
7. Predictions are averaged over the last **45 frames (~1.5 seconds)**.
8. The result is displayed in real time.

---

## Extracted Features

- Head tilt angle
- Shoulder slope
- Nose distance from shoulder midpoint
- Left elbow (X, Y, Z, distance)
- Right elbow (X, Y, Z, distance)
- Shoulder asymmetry
- Head rotation
- Elbow-to-elbow distance

---

# 📂 Project Structure

```text
PostureAI/
│
├── posture_check.py
├── model/
│   ├── posture_model_optuna.pkl
│   ├── scaler_optuna.pkl
│   └── pose_landmarker_lite.task
├── mediapipe.ipynb
├── model.ipynb
├── requirements.txt
├── LICENSE
└── README.md
```

---

# 📊 Model Performance

The classifier was trained using the **Multi-Gait & Posture** dataset from **PhysioNet**.

| Metric | Value |
|---------|------:|
| Validation F1 Score | **0.9229** |
| Classes | Good (0), Bad (1) |

Hyperparameters were optimized using **Optuna**, resulting in robust performance across different users and lighting conditions.

---

# 🔮 Future Improvements

- Audio alerts for poor posture
- Statistics dashboard
- Session history
- Multiple posture categories
- Cross-platform desktop application
- Mobile support

---

# 📄 License

This project is licensed under the **MIT License**.

---

## ⭐ Support

If you found this project useful, consider giving it a **⭐ Star** on GitHub.
