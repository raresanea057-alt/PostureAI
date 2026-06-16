import streamlit as st
import numpy as np
import joblib
import cv2
from PIL import Image
from mediapipe import Image as MpImage, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    vec = nose - shoulder_mid
    vertical = np.array([0, 1, 0])
    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-8)
    f.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    shoulder_vec = right_shoulder - left_shoulder
    horizontal = np.array([1, 0, 0])
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

    vec_proj = vec.copy(); vec_proj[1] = 0
    shoulder_proj = shoulder_vec.copy(); shoulder_proj[1] = 0
    if np.linalg.norm(vec_proj) > 0 and np.linalg.norm(shoulder_proj) > 0:
        cos_rot = np.dot(vec_proj, shoulder_proj) / (np.linalg.norm(vec_proj) * np.linalg.norm(shoulder_proj) + 1e-8)
        f.append(np.degrees(np.arccos(np.clip(cos_rot, -1.0, 1.0))))
    else:
        f.append(0.0)

    f.append(np.linalg.norm(left_elbow - right_elbow) / shoulder_width)
    return np.array(f).reshape(1, -1)

@st.cache_resource
def load_resources():
    model  = joblib.load("posture_model_optuna.pkl")
    scaler = joblib.load("scaler_optuna.pkl")
    good_idx = list(model.classes_).index(0)
    base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    return model, scaler, good_idx, detector

st.set_page_config(page_title="PostureAI", page_icon="🧍")
st.title("🧍 PostureAI")
st.caption("Upload a photo to check your posture — MediaPipe + Random Forest")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    model, scaler, good_idx, detector = load_resources()

    img = Image.open(uploaded).convert("RGB")
    frame = np.array(img)
    mp_img = MpImage(image_format=ImageFormat.SRGB, data=frame)
    result = detector.detect(mp_img)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        feat = compute_features(landmarks)
        feat_scaled = scaler.transform(feat)
        prob_good = model.predict_proba(feat_scaled)[0][good_idx]

        label = "✅ Good Posture" if prob_good > 0.3 else "❌ Bad Posture"
        color = "green" if prob_good > 0.3 else "red"

        st.image(img, use_container_width=True)
        st.markdown(f"### :{color}[{label}]")
        st.metric("Confidence", f"{prob_good:.0%}")
    else:
        st.image(img, use_container_width=True)
        st.warning("No pose detected. Make sure your upper body is visible.")
