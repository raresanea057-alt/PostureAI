# Generated from: app.ipynb
# Converted at: 2026-06-16T18:38:50.634Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# app.py
import cv2
import numpy as np
import joblib
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from collections import deque
import urllib.request
import os
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Download model MediaPipe ─────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

# ── Constante landmarks ───────────────────────────────────────────────────────
NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW = 0, 11, 12, 13, 14

def get_pt(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])

def compute_features(landmarks):
    nose           = get_pt(landmarks, NOSE)
    left_shoulder  = get_pt(landmarks, LEFT_SHOULDER)
    right_shoulder = get_pt(landmarks, RIGHT_SHOULDER)
    left_elbow     = get_pt(landmarks, LEFT_ELBOW)
    right_elbow    = get_pt(landmarks, RIGHT_ELBOW)

    shoulder_mid   = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    f = []
    vec      = nose - shoulder_mid
    vertical = np.array([0, 1, 0])
    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-8)
    f.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    shoulder_vec = right_shoulder - left_shoulder
    horizontal   = np.array([1, 0, 0])
    cos_slope = np.dot(shoulder_vec, horizontal) / (np.linalg.norm(shoulder_vec) + 1e-8)
    f.append(np.degrees(np.arccos(np.clip(cos_slope, -1.0, 1.0))))

    f.append(np.linalg.norm(nose - shoulder_mid) / shoulder_width)
    f.append((left_elbow[2]  - left_shoulder[2])  / shoulder_width)
    f.append((left_elbow[0]  - left_shoulder[0])  / shoulder_width)
    f.append((left_elbow[1]  - left_shoulder[1])  / shoulder_width)
    f.append(np.linalg.norm(left_elbow - left_shoulder) / shoulder_width)
    f.append((right_elbow[2] - right_shoulder[2]) / shoulder_width)
    f.append((right_elbow[0] - right_shoulder[0]) / shoulder_width)
    f.append((right_elbow[1] - right_shoulder[1]) / shoulder_width)
    f.append(np.linalg.norm(right_elbow - right_shoulder) / shoulder_width)
    f.append((left_shoulder[1] - right_shoulder[1]) / shoulder_width)

    vec_proj      = vec.copy();          vec_proj[1] = 0
    shoulder_proj = shoulder_vec.copy(); shoulder_proj[1] = 0
    if np.linalg.norm(vec_proj) > 0 and np.linalg.norm(shoulder_proj) > 0:
        cos_rot = np.dot(vec_proj, shoulder_proj) / (np.linalg.norm(vec_proj) * np.linalg.norm(shoulder_proj) + 1e-8)
        f.append(np.degrees(np.arccos(np.clip(cos_rot, -1.0, 1.0))))
    else:
        f.append(0.0)

    f.append(np.linalg.norm(left_elbow - right_elbow) / shoulder_width)
    return np.array(f).reshape(1, -1)

# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model  = joblib.load('posture_model_optuna.pkl')
    scaler = joblib.load('scaler_optuna.pkl')
    assert list(model.classes_) == [0, 1], f"Unexpected class order: {model.classes_}"
    good_idx = list(model.classes_).index(0)
    return model, scaler, good_idx

@st.cache_resource
def load_detector():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    return vision.PoseLandmarker.create_from_options(options)

# ── Video processor ───────────────────────────────────────────────────────────
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.scaler, self.good_idx = load_model()
        self.detector    = load_detector()
        self.prob_buffer = deque(maxlen=15)  # ~0.5s la 30fps

        CONNECTIONS = [
            (0,1),(0,4),(1,2),(2,3),(4,5),(5,6),(7,0),(8,0),
            (9,0),(10,0),(11,12),(11,13),(12,14)
        ]
        self.CONNECTIONS     = CONNECTIONS
        self.LANDMARK_INDICES = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW]

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            feat        = compute_features(landmarks)
            feat_scaled = self.scaler.transform(feat)
            prob_good   = self.model.predict_proba(feat_scaled)[0][self.good_idx]

            self.prob_buffer.append(prob_good)
            smoothed = np.mean(self.prob_buffer)

            pred  = 0 if smoothed > 0.3 else 1
            label = "Good" if pred == 0 else "Bad"
            color = (0, 255, 0) if pred == 0 else (0, 0, 255)

            cv2.putText(img, f"Posture: {label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(img, f"Prob: {smoothed:.2f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            h, w, _ = img.shape
            for idx in self.LANDMARK_INDICES:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, color, -1)
            for (i, j) in self.CONNECTIONS:
                lm1, lm2 = landmarks[i], landmarks[j]
                x1, y1 = int(lm1.x * w), int(lm1.y * h)
                x2, y2 = int(lm2.x * w), int(lm2.y * h)
                cv2.line(img, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.putText(img, "No pose detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="PostureAI", page_icon="🧍", layout="centered")
st.title("🧍 PostureAI")
st.caption("Real-time posture detection via webcam — MediaPipe + Random Forest")
st.divider()

webrtc_streamer(
    key="posture",
    video_processor_factory=PostureProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

import sys
import subprocess

subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
