import streamlit as st
import cv2
import numpy as np
import joblib
import os
import urllib.request
import time
from collections import deque
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PostureAI",
    page_icon="🧍",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0d0f14; }

  .posture-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #e0f7ff 0%, #7dd3fc 50%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
  }
  .posture-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 4px;
    font-weight: 400;
  }

  .status-good {
    display: inline-block;
    background: #052e16;
    color: #4ade80;
    border: 1px solid #16a34a;
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.04em;
  }
  .status-bad {
    display: inline-block;
    background: #2c0614;
    color: #f87171;
    border: 1px solid #dc2626;
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
    letter-spacing: 0.04em;
  }
  .status-none {
    display: inline-block;
    background: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 999px;
    padding: 6px 18px;
    font-size: 1rem;
    font-weight: 600;
    font-family: 'Space Grotesk', sans-serif;
  }

  .metric-card {
    background: #131720;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 16px 20px;
  }
  .metric-label {
    color: #475569;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }
  .metric-value {
    color: #e2e8f0;
    font-size: 1.5rem;
    font-weight: 700;
    font-family: 'Space Grotesk', sans-serif;
  }

  .prob-bar-bg {
    background: #1e293b;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 8px;
  }

  section[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1e293b;
  }

  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

NOSE           = 0
LEFT_SHOULDER  = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW     = 13
RIGHT_ELBOW    = 14
LANDMARK_INDICES = [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_ELBOW, RIGHT_ELBOW]

CONNECTIONS = [
    (0,1),(0,4),(1,2),(2,3),(4,5),(5,6),
    (7,0),(8,0),(9,0),(10,0),
    (11,12),(11,13),(12,14)
]

CALIB_SECONDS = 5
BUFFER_SIZE   = 45  # ~1.5s at 30fps

# ── Cached resource loading ────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    if not os.path.exists(MODEL_PATH):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    clf      = joblib.load("posture_model_optuna.pkl")
    scaler   = joblib.load("scaler_optuna.pkl")
    return detector, clf, scaler

# ── Landmark smoothing (from notebook) ────────────────────────────────────────
_prev_landmarks = None
ALPHA = 0.7

def smooth_landmarks(landmarks):
    global _prev_landmarks
    if _prev_landmarks is None:
        _prev_landmarks = landmarks
        return landmarks
    smoothed = []
    for i in range(len(landmarks)):
        x = ALPHA * _prev_landmarks[i].x + (1 - ALPHA) * landmarks[i].x
        y = ALPHA * _prev_landmarks[i].y + (1 - ALPHA) * landmarks[i].y
        z = ALPHA * _prev_landmarks[i].z + (1 - ALPHA) * landmarks[i].z
        smoothed.append(type(landmarks[i])(x, y, z))
    _prev_landmarks = smoothed
    return smoothed

# ── Feature extraction ─────────────────────────────────────────────────────────
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

    vertical  = np.array([0, 1, 0])
    cos_angle = np.dot(vec, vertical) / (np.linalg.norm(vec) * np.linalg.norm(vertical) + 1e-8)
    f.append(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))

    shoulder_vec = right_shoulder - left_shoulder
    horizontal   = np.array([1, 0, 0])
    cos_slope    = np.dot(shoulder_vec, horizontal) / (np.linalg.norm(shoulder_vec) * np.linalg.norm(horizontal) + 1e-8)
    f.append(np.degrees(np.arccos(np.clip(cos_slope, -1.0, 1.0))))

    f.append(np.linalg.norm(nose - shoulder_mid) / shoulder_width)

    f.append((left_elbow[2]  - left_shoulder[2])  / shoulder_width)
    f.append((left_elbow[0]  - left_shoulder[0])  / shoulder_width)
    f.append((left_elbow[1]  - left_shoulder[1])  / shoulder_width)
    f.append(np.linalg.norm(left_elbow  - left_shoulder)  / shoulder_width)

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

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="posture-title">PostureAI</p>', unsafe_allow_html=True)
st.markdown('<p class="posture-sub">Real-time posture detection · MediaPipe Pose + Random Forest</p>', unsafe_allow_html=True)
st.markdown("---")

col_cam, col_side = st.columns([3, 1], gap="large")

with col_side:
    st.markdown("#### Settings")
    threshold = st.slider("Good-posture threshold", 0.0, 1.0, 0.50, 0.01,
                          help="Auto-set after calibration. Lower = stricter.")
    st.markdown("---")
    st.markdown("#### Status")
    status_placeholder = st.empty()
    st.markdown("---")
    st.markdown("#### Probabilities")
    smooth_placeholder = st.empty()
    inst_placeholder   = st.empty()
    st.markdown("---")
    st.caption("Models: `posture_model_optuna.pkl` · `scaler_optuna.pkl`")
    st.caption("Landmark indices 0–14 (no wrists)")

with col_cam:
    start_btn = st.button("▶ Start Camera", type="primary")
    frame_placeholder = st.empty()

# ── Camera loop ────────────────────────────────────────────────────────────────
if start_btn:
    global _prev_landmarks
    _prev_landmarks = None  # reset smoothing state

    detector, clf, scaler = load_models()
    GOOD_CLASS_IDX = list(clf.classes_).index(0)

    cap          = cv2.VideoCapture(0)
    prob_buffer  = deque(maxlen=BUFFER_SIZE)
    calib_frames = []
    calibrating  = True
    calib_start  = time.time()
    calib_mean   = None
    active_threshold = threshold

    status_placeholder.markdown('<span class="status-none">— Calibrating…</span>', unsafe_allow_html=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_img)

        # ── CALIBRATION PHASE ────────────────────────────────────────────────
        if calibrating:
            elapsed   = time.time() - calib_start
            remaining = max(0, int(CALIB_SECONDS - elapsed))

            if result.pose_landmarks:
                landmarks = smooth_landmarks(result.pose_landmarks[0])
                feat = compute_features(landmarks)
                calib_frames.append(feat[0])

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
                    calib_mean  = np.mean(calib_frames, axis=0)
                    raw_mean    = calib_mean.reshape(1, -1)
                    scaled_mean = scaler.transform(raw_mean)
                    calib_prob  = clf.predict_proba(scaled_mean)[0][GOOD_CLASS_IDX]
                    active_threshold = float(np.clip(calib_prob - 0.15, 0.2, 0.85))
                else:
                    calib_mean = np.zeros(14)

            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            continue

        # ── INFERENCE PHASE ──────────────────────────────────────────────────
        label  = "No pose"
        smooth = 0.0
        inst   = 0.0
        color  = (80, 80, 200)

        if result.pose_landmarks:
            landmarks  = smooth_landmarks(result.pose_landmarks[0])
            feat       = compute_features(landmarks)

            feat_delta = feat - calib_mean
            feat_final = np.concatenate([feat, feat_delta])   # (2,14) → scaler sees row 0
            feat_scaled = scaler.transform(feat_final)
            prob_good  = clf.predict_proba(feat_scaled)[0][GOOD_CLASS_IDX]

            prob_buffer.append(prob_good)
            smooth = float(np.mean(prob_buffer))
            inst   = float(prob_good)

            pred  = 0 if smooth > active_threshold else 1
            label = "Good" if pred == 0 else "Bad"
            color = (0, 220, 80) if pred == 0 else (0, 0, 220)

            h, w = frame.shape[:2]
            for idx in LANDMARK_INDICES:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, color, -1)
                cv2.circle(frame, (cx, cy), 7, (255, 255, 255), 1)
            for (i, j) in CONNECTIONS:
                lm1, lm2 = landmarks[i], landmarks[j]
                cv2.line(frame,
                         (int(lm1.x * w), int(lm1.y * h)),
                         (int(lm2.x * w), int(lm2.y * h)),
                         color, 2)

            overlay = frame.copy()
            cv2.rectangle(overlay, (8, 8), (320, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
            cv2.putText(frame, f"Posture: {label}", (16, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Smooth: {smooth:.2f}  inst: {inst:.2f}", (16, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No pose detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # ── Update sidebar ───────────────────────────────────────────────────
        if label == "Good":
            status_placeholder.markdown('<span class="status-good">✓ Good posture</span>', unsafe_allow_html=True)
        elif label == "Bad":
            status_placeholder.markdown('<span class="status-bad">⚠ Bad posture</span>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<span class="status-none">— No pose</span>', unsafe_allow_html=True)

        bar_color = "#4ade80" if label == "Good" else "#f87171"

        smooth_placeholder.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Smoothed prob</div>
          <div class="metric-value">{smooth:.2f}</div>
          <div class="prob-bar-bg">
            <div style="height:6px;border-radius:4px;width:{int(smooth*100)}%;background:{bar_color};transition:width 0.3s"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        inst_placeholder.markdown(f"""
        <div class="metric-card" style="margin-top:8px">
          <div class="metric-label">Instant prob</div>
          <div class="metric-value">{inst:.2f}</div>
          <div class="prob-bar-bg">
            <div style="height:6px;border-radius:4px;width:{int(inst*100)}%;background:#38bdf8;transition:width 0.3s"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    cap.release()
    detector.close()
    frame_placeholder.empty()
    status_placeholder.markdown('<span class="status-none">— Stopped</span>', unsafe_allow_html=True)
