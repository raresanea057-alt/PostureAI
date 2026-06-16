<div align="center">
🧍 PostureAI

Real-time posture detection via webcam — built with Python & MediaPipe

Show Image
Show Image
Show Image
Show Image
Show Image

</div>

About

PostureAI uses your webcam and MediaPipe's pose estimation to detect bad posture in real time. It tracks key body landmarks and alerts you when it detects slouching, forward head position, or shoulder imbalance — right on your screen as it happens.

No wearables. No subscriptions. Just your webcam.


Features


🎯 Real-time pose detection via MediaPipe Pose landmarks
⚠️ Bad posture alerts — slouch, forward head, uneven shoulders
🖥️ On-screen feedback overlaid on the live webcam feed
⚡ Lightweight — runs locally, no data sent anywhere



Getting Started

Prerequisites


Python 3.10+
A webcam


Installation

bashgit clone https://github.com/yourusername/PostureAI.git
cd PostureAI
pip install -r requirements.txt

Run

bashpython main.py

Press Q to quit.


Project Structure

PostureAI/
├── main.py           # Entry point — webcam loop
├── detector.py       # MediaPipe pose detection
├── feedback.py       # Posture analysis & alert logic
├── requirements.txt
└── README.md


Roadmap


 Real-time pose detection
 Slouch & forward head detection
 On-screen posture score
 Session history & progress tracking
 Notification alerts
 Dashboard UI



Contributing

Pull requests are welcome. For major changes, open an issue first.


License

MIT
