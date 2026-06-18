import streamlit as st
import cv2
import numpy as np
import joblib
import os
import urllib.request
from collections import deque
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

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

  /* Hero title */
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

  /* Status pill */
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

  /* Metric card */
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

  /* Probability bar */
  .prob-bar-bg {
    background: #1e293b;
    border-radius: 4px;
    height: 6px;
    width: 100%;
    margin-top: 8px;
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background: #0d0f14;
    border-right: 1px solid #1e293b;
  }

  /* Hide Streamlit branding */
  #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_PATH = "pose_landmarker_lite.task"

LANDMARK_INDICES = list(range(15))
CONNECTIONS = [
    (0,1),(0,4),(1,2),(2,3),(4,5),(5,6),
    (7,0),(8,0),(9,0),(10,0),
    (11,12),(11,13),(12,14)
]

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

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
    clf     = joblib.load("posture_model_optuna.pkl")
    scaler  = joblib.load("scaler_optuna.pkl")
    return detector, clf, scaler

# ── Feature extraction ─────────────────────────────────────────────────────────
def compute_features(landmarks):
    pts           = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in LANDMARK_INDICES])
    nose          = pts[0]
    left_shoulder = pts[11]; right_shoulder = pts[12]
    left_elbow    = pts[13]; right_elbow    = pts[14]

    shoulder_mid   = (left_shoulder + right_shoulder) / 2.0
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    if shoulder_width < 1e-6:
        shoulder_width = 1.0

    f   = []
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

# ── Video processor ────────────────────────────────────────────────────────────
class PostureProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector, self.clf, self.scaler = load_models()
        self.prob_buffer  = deque(maxlen=15)   # ~0.5 s at 30 fps
        self.threshold    = 0.30
        self.last_label   = "No pose"
        self.last_prob    = 0.0
        self.last_smooth  = 0.0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(mp_img)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks[0]
            feat       = compute_features(landmarks)
            feat_s     = self.scaler.transform(feat)
            prob_good  = self.clf.predict_proba(feat_s)[0][0]

            self.prob_buffer.append(prob_good)
            smoothed          = float(np.mean(self.prob_buffer))
            pred              = 0 if smoothed > self.threshold else 1
            self.last_label   = "Good" if pred == 0 else "Bad"
            self.last_prob    = float(prob_good)
            self.last_smooth  = smoothed

            # ── draw skeleton ───────────────────────────────────────────────
            h, w, _ = img.shape
            color    = (0, 220, 80) if pred == 0 else (60, 60, 255)

            for idx in LANDMARK_INDICES:
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(img, (cx, cy), 5, color, -1)
                cv2.circle(img, (cx, cy), 7, (255, 255, 255), 1)

            for (i, j) in CONNECTIONS:
                lm1, lm2 = landmarks[i], landmarks[j]
                x1, y1   = int(lm1.x * w), int(lm1.y * h)
                x2, y2   = int(lm2.x * w), int(lm2.y * h)
                cv2.line(img, (x1, y1), (x2, y2), color, 2)

            # ── HUD overlay ─────────────────────────────────────────────────
            overlay = img.copy()
            cv2.rectangle(overlay, (8, 8), (320, 80), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

            label_text = f"Posture: {self.last_label}"
            cv2.putText(img, label_text, (16, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
            cv2.putText(img, f"Smooth: {smoothed:.2f}  inst: {prob_good:.2f}", (16, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        else:
            self.last_label  = "No pose"
            self.last_prob   = 0.0
            self.last_smooth = 0.0
            cv2.putText(img, "No pose detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 200), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Layout ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="posture-title">PostureAI</p>', unsafe_allow_html=True)
st.markdown('<p class="posture-sub">Real-time posture detection · MediaPipe Pose + Random Forest</p>', unsafe_allow_html=True)
st.markdown("---")

col_cam, col_side = st.columns([3, 1], gap="large")

with col_side:
    st.markdown("#### Settings")
    threshold = st.slider("Good-posture threshold", 0.0, 1.0, 0.30, 0.01,
                          help="Lower = stricter. Probability of 'Good' must exceed this.")
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
    ctx = webrtc_streamer(
        key="posture-ai",
        video_processor_factory=PostureProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ── Live status panel (updates while stream is active) ─────────────────────────
if ctx.video_processor:
    ctx.video_processor.threshold = threshold   # push slider value

    import time
    while ctx.state.playing:
        proc = ctx.video_processor
        label   = proc.last_label
        smooth  = proc.last_smooth
        inst    = proc.last_prob

        if label == "Good":
            status_placeholder.markdown('<span class="status-good">✓ Good posture</span>', unsafe_allow_html=True)
        elif label == "Bad":
            status_placeholder.markdown('<span class="status-bad">⚠ Bad posture</span>', unsafe_allow_html=True)
        else:
            status_placeholder.markdown('<span class="status-none">— No pose</span>', unsafe_allow_html=True)

        bar_w_smooth = int(smooth * 100)
        bar_w_inst   = int(inst   * 100)
        bar_color    = "#4ade80" if label == "Good" else "#f87171"

        smooth_placeholder.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">Smoothed prob</div>
          <div class="metric-value">{smooth:.2f}</div>
          <div class="prob-bar-bg">
            <div style="height:6px;border-radius:4px;width:{bar_w_smooth}%;background:{bar_color};transition:width 0.3s"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        inst_placeholder.markdown(f"""
        <div class="metric-card" style="margin-top:8px">
          <div class="metric-label">Instant prob</div>
          <div class="metric-value">{inst:.2f}</div>
          <div class="prob-bar-bg">
            <div style="height:6px;border-radius:4px;width:{bar_w_inst}%;background:#38bdf8;transition:width 0.3s"></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        time.sleep(0.15)
