# config.py

import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))

# ── Database path — uses env var in Docker, local file otherwise ──────────────
DATABASE_PATH = os.environ.get(
    "DATABASE_PATH",
    os.path.join(BASE_DIR, "yoga_app.db")
)


DATASET_DIR = os.path.join(BASE_DIR, "dataset", "yoga82")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
STATIC_DIR    = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# ─── Dataset ────────────────────────────────────────────────────────────────
NUM_CLASSES   = 82
IMAGE_SIZE = (320, 240)   # ← reduced from (224, 224)

# ─── Model paths (.pkl instead of .h5) ──────────────────────────────────────
MODEL_PATH    = os.path.join(MODELS_DIR, "yoga_pose_model.pkl")
LABELS_PATH   = os.path.join(MODELS_DIR, "class_labels.json")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")   # for feature normalization

# ─── ML model settings ──────────────────────────────────────────────────────
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,      # number of trees
    "max_depth": None,        # let trees grow fully
    "min_samples_split": 2,
    "random_state": 42,
    "n_jobs": -1,             # use all CPU cores
}
SVM_PARAMS = {
    "kernel": "rbf",
    "C": 10,
    "gamma": "scale",
    "probability": True,      # needed to get confidence scores
}

# ─── Pose correctness ────────────────────────────────────────────────────────
ANGLE_THRESHOLD  = 15         # degrees
MIN_CONFIDENCE   = 0.70

# ─── Calorie calculation ─────────────────────────────────────────────────────
DEFAULT_WEIGHT_KG = 65.0
MET_VALUES = {
    "standing": 2.5, "seated": 2.0, "supine": 1.5,
    "prone": 1.5, "balancing": 3.0, "inverted": 3.5,
    "backbend": 3.0, "forward_bend": 2.5, "twist": 2.0,
    "default": 2.5,
}

# ─── Flask ───────────────────────────────────────────────────────────────────
SECRET_KEY = "yoga_ml_secret_key"
DEBUG      = True
HOST       = "0.0.0.0"
PORT       = 5000

# ─── MediaPipe ───────────────────────────────────────────────────────────────
MP_DETECTION_CONFIDENCE = 0.3
MP_TRACKING_CONFIDENCE  = 0.5

# ─── Feature engineering ─────────────────────────────────────────────────────
# Joint angle triplets: (point_A, vertex, point_B)
# These are MediaPipe landmark indices (0–32)
ANGLE_TRIPLETS = [
    (11, 13, 15),   # left elbow
    (12, 14, 16),   # right elbow
    (13, 11, 23),   # left shoulder
    (14, 12, 24),   # right shoulder
    (11, 23, 25),   # left hip
    (12, 24, 26),   # right hip
    (23, 25, 27),   # left knee
    (24, 26, 28),   # right knee
    (25, 27, 29),   # left ankle
    (26, 28, 30),   # right ankle
    (11, 12, 24),   # shoulder width angle
    (23, 24, 26),   # hip width angle
]

# ─── Pose validation thresholds ──────────────────────────────────────────────
MIN_HOLD_SECONDS  = 3      # minimum seconds user must hold a pose
MIN_POSE_SCORE    = 60.0   # minimum correctness score to count calories
MIN_CONFIDENCE    = 0.55   # minimum model confidence to accept prediction