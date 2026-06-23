#!/usr/bin/env python3
"""
Real‑time posture detection using MediaPipe Pose and a trained classifier.
Calibrates for 5 seconds to set a personal baseline, then continuously
classifies posture as "Good" or "Bad".
"""

import os
import time
import urllib.request
from collections import deque

import cv2
import numpy as np
import joblib
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------------------------------------------------
#  Constants & paths
# ----------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = os.path.join(BASE, "pose_landmarker_lite.task")

# Landmark indices used for feature extraction
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
RIGHT_ELBOW = 14

LANDMARK_INDICES = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW]

# Skeleton connections for drawing
CONNECTIONS = [
    (0, 1), (0, 4), (1, 2), (2, 3), (4, 5), (5, 6),
    (7, 0), (8, 0), (9, 0), (10, 0), (11, 12), (11, 13), (12, 14)
]

# ----------------------------------------------------------------------
#  Load models
# ----------------------------------------------------------------------
print("Loading posture model and scaler...")
BASE = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE, 'model', 'posture_model_optuna.pkl'))
scaler = joblib.load(os.path.join(BASE, "model", "scaler_optuna.pkl"))

# Ensure class order (0 = Good, 1 = Bad)
if list(model.classes_) != [0, 1]:
    raise ValueError(f"Unexpected class order: {model.classes_}")
GOOD_CLASS_IDX = 0

# ----------------------------------------------------------------------
#  Download MediaPipe pose model (if missing)
# ----------------------------------------------------------------------
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading pose landmarker model (~10 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("✅ Downloaded")

# ----------------------------------------------------------------------
#  Feature extraction
# ----------------------------------------------------------------------
def get_pt(landmarks, idx):
    lm = landmarks[idx]
    return np.array([lm.x, lm.y, lm.z])

def compute_features(landmarks):
    """Extract 14 geometric features from pose landmarks."""
    nose = get_pt(landmarks, NOSE)
    ls = get_pt(landmarks, LEFT_SHOULDER)
    rs = get_pt(landmarks, RIGHT_SHOULDER)
    le = get_pt(landmarks, LEFT_ELBOW)
    re = get_pt(landmarks, RIGHT_ELBOW)

    shoulder_mid = (ls + rs) / 2.0
    shoulder_width = np.linalg.norm(ls - rs)
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    f = []

    # 1. Head tilt (angle between nose->shoulder_mid and vertical)
    vec = nose - shoulder_mid
    vertical = np.array([0, 1, 0])
    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-8)
    f.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    # 2. Shoulder slope (angle between shoulder line and horizontal)
    shoulder_vec = rs - ls
    horizontal = np.array([1, 0, 0])
    cos_slope = np.dot(shoulder_vec, horizontal) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(horizontal) + 1e-8)
    f.append(np.degrees(np.arccos(np.clip(cos_slope, -1.0, 1.0))))

    # 3. Nose distance from shoulder midpoint (normalised)
    f.append(np.linalg.norm(nose - shoulder_mid) / shoulder_width)

    # 4-7. Left elbow relative to left shoulder (z, x, y, distance)
    f.append((le[2] - ls[2]) / shoulder_width)
    f.append((le[0] - ls[0]) / shoulder_width)
    f.append((le[1] - ls[1]) / shoulder_width)
    f.append(np.linalg.norm(le - ls) / shoulder_width)

    # 8-11. Right elbow relative to right shoulder (z, x, y, distance)
    f.append((re[2] - rs[2]) / shoulder_width)
    f.append((re[0] - rs[0]) / shoulder_width)
    f.append((re[1] - rs[1]) / shoulder_width)
    f.append(np.linalg.norm(re - rs) / shoulder_width)

    # 12. Shoulder asymmetry (y-axis difference)
    f.append((ls[1] - rs[1]) / shoulder_width)

    # 13. Head rotation (projected onto horizontal plane)
    vec_proj = vec.copy()
    vec_proj[1] = 0
    shoulder_proj = shoulder_vec.copy()
    shoulder_proj[1] = 0
    if np.linalg.norm(vec_proj) > 0 and np.linalg.norm(shoulder_proj) > 0:
        cos_rot = np.dot(vec_proj, shoulder_proj) / (np.linalg.norm(vec_proj) * np.linalg.norm(shoulder_proj) + 1e-8)
        f.append(np.degrees(np.arccos(np.clip(cos_rot, -1.0, 1.0))))
    else:
        f.append(0.0)

    # 14. Elbow-elbow distance
    f.append(np.linalg.norm(le - re) / shoulder_width)

    return np.array(f).reshape(1, -1)

# ----------------------------------------------------------------------
#  Smoothing (exponential moving average)
# ----------------------------------------------------------------------
ALPHA = 0.7
prev_landmarks = None

def smooth_landmarks(landmarks):
    global prev_landmarks
    if prev_landmarks is None:
        prev_landmarks = landmarks
        return landmarks

    smoothed = []
    for i in range(len(landmarks)):
        x = ALPHA * prev_landmarks[i].x + (1 - ALPHA) * landmarks[i].x
        y = ALPHA * prev_landmarks[i].y + (1 - ALPHA) * landmarks[i].y
        z = ALPHA * prev_landmarks[i].z + (1 - ALPHA) * landmarks[i].z
        smoothed.append(type(landmarks[i])(x, y, z))
    prev_landmarks = smoothed
    return smoothed

# ----------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------
def main():
    # Create MediaPipe pose detector
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
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Calibration parameters
    CALIB_SECONDS = 5
    BUFFER_SIZE = 45          # 1.5 s at 30 fps
    BASE_THRESHOLD = 0.5

    prob_buffer = deque(maxlen=BUFFER_SIZE)
    calib_frames = []
    calibrating = True
    calib_start = time.time()
    threshold = BASE_THRESHOLD
    calib_mean = None

    print(f"CALIBRATION: Sit in your best posture for {CALIB_SECONDS} seconds...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        # ------------------------------------------------------------------
        #  Calibration phase
        # ------------------------------------------------------------------
        if calibrating:
            elapsed = time.time() - calib_start
            remaining = max(0, int(CALIB_SECONDS - elapsed))

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                landmarks = smooth_landmarks(landmarks)
                feat = compute_features(landmarks)      # shape (1,14)
                calib_frames.append(feat[0])

            # Overlay dark background with calibration text
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

            h, w = frame.shape[:2]
            cv2.putText(frame, "CALIBRATING", (w//2 - 160, h//2 - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 220, 80), 3, cv2.LINE_AA)
            cv2.putText(frame, "Sit in your best posture", (w//2 - 175, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
            cv2.putText(frame, f"{remaining}s", (w//2 - 35, h//2 + 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 220, 80), 4, cv2.LINE_AA)

            if elapsed >= CALIB_SECONDS:
                calibrating = False
                if len(calib_frames) > 10:
                    # Compute mean feature vector (for reference, not used in prediction)
                    calib_mean = np.mean(calib_frames, axis=0)
                    # Calculate personal good posture probability (just for info)
                    scaled_mean = scaler.transform(calib_mean.reshape(1, -1))
                    calib_prob = model.predict_proba(scaled_mean)[0][GOOD_CLASS_IDX]
                    threshold = BASE_THRESHOLD
                    print(f"Calibration done. Your good posture score: {calib_prob:.2f} → threshold set to {threshold:.2f}")
                else:
                    print("Calibration: no pose detected. Using default threshold.")
                    calib_mean = None
                    threshold = BASE_THRESHOLD

            cv2.imshow('Posture Check', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # ------------------------------------------------------------------
        #  Inference phase
        # ------------------------------------------------------------------
        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            feat = compute_features(landmarks)           # shape (1,14)

            # Scale features and predict
            feat_scaled = scaler.transform(feat)
            prob_good = model.predict_proba(feat_scaled)[0][GOOD_CLASS_IDX]
            prob_buffer.append(prob_good)
            smoothed_prob = np.mean(prob_buffer)

            pred = 0 if smoothed_prob > threshold else 1
            label = "Good" if pred == 0 else "Bad"
            color = (0, 220, 80) if pred == 0 else (0, 0, 220)

            # Draw info
            cv2.putText(frame, f"Posture: {label}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Smooth: {smoothed_prob:.2f}  inst: {prob_good:.2f}", (10, 85),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw landmarks and skeleton
            h, w = frame.shape[:2]
            for idx in LANDMARK_INDICES:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
            for (i, j) in CONNECTIONS:
                lm1, lm2 = landmarks[i], landmarks[j]
                cv2.line(frame,
                         (int(lm1.x * w), int(lm1.y * h)),
                         (int(lm2.x * w), int(lm2.y * h)),
                         color, 2)
        else:
            cv2.putText(frame, "No pose detected", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 200), 2, cv2.LINE_AA)

        cv2.imshow('Posture Check', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()