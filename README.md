# 🧍 PostureAI

<div align="center">

### Real-Time Posture Detection with Computer Vision

Detect slouching, forward head posture, and shoulder imbalance directly from your webcam using **Python**, **MediaPipe**, and a trained classifier.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-FF6F00)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📸 Demo

<p align="center">
  <img src="assets/demo.gif" width="850" alt="PostureAI Demo">
</p>

---

## 🚀 About

PostureAI is a real-time posture monitoring application that uses your webcam to track body landmarks via **MediaPipe Pose**. It extracts geometric features (head tilt, shoulder slope, elbow positions, etc.) and feeds them into a **trained classifier** to instantly tell you if your posture is **Good** or **Bad**.

No wearables, no cloud – everything runs **100% locally**.

---

## ✨ Features

- 🎯 Real-time pose tracking with MediaPipe
- 🧘 **5‑second calibration** – sit in your best posture to set a personal baseline
- 📊 **Smoothed predictions** to avoid jittery feedback
- 🖥️ Live visual feedback – landmarks, skeleton, and status text
- ⚠️ Detects:
  - Slouching (forward head posture)
  - Shoulder asymmetry / imbalance
  - Poor elbow alignment
- 🔒 All processing stays on your machine

---

## 🛠️ Built With

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- scikit-learn (for model loading)
- Joblib (for model serialization)

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/raresanea057-alt/PostureAI.git
cd PostureAI
