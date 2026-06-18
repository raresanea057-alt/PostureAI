# 🧍 PostureAI

<div align="center">

### Real-Time Posture Detection with Computer Vision

Detect slouching, forward head posture, and shoulder imbalance directly from your webcam using **Python**, **OpenCV**, and **MediaPipe**.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python\&logoColor=white)
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


## Model

Trained on the [Multi-Gait & Posture dataset](https://physionet.org/content/multi-gait-posture/1.0.0/) (PhysioNet).  
Achieved **F1 score of 0.9229** on the validation set.

## 🚀 About

PostureAI is a real-time posture monitoring application that analyzes body posture through a webcam feed.

Using MediaPipe Pose Estimation, the system tracks body landmarks and provides instant feedback whenever poor posture is detected.

No wearables. No subscriptions. Just your webcam.

---

## ✨ Features

* 🎯 Real-time pose tracking
* ⚠️ Slouch detection 
* ⚠️ ### <TODO>Forward head posture detection
* ⚠️ ### <TODO> Shoulder imbalance detection 
* 🖥️ Live visual feedback
* 🔒 100% local processing
* ⚡ Lightweight and fast

---

## 🛠️ Built With

* Python
* OpenCV
* MediaPipe
* NumPy

---

## 📦 Installation

### Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/PostureAI.git
cd PostureAI
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the application

```bash
python main.py
```

Press **Q** to quit.

---

## 📂 Project Structure

```text
PostureAI/
│
├── main.py          # Application entry point
├── detector.py      # Pose landmark detection
├── feedback.py      # Posture analysis logic
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

1. Capture webcam frames
2. Extract pose landmarks using MediaPipe
3. Calculate posture metrics
4. Detect posture issues
5. Display live feedback

---

## 🗺️ Roadmap

* [x] Real-time pose detection
* [x] Slouch detection
* [x] Forward head posture detection
* [x] Shoulder alignment monitoring
* [ ] Posture score
* [ ] Session analytics
* [ ] Progress tracking
* [ ] Desktop notifications
* [ ] Dashboard UI

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License.

---

<div align="center">

⭐ If you find this project useful, consider giving it a star.

</div>
