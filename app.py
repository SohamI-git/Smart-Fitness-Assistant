# app.py

import os
import cv2
import json
import time
import base64
import numpy as np
import joblib
import mediapipe as mp
from datetime       import datetime
from flask          import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_login    import LoginManager, login_required, current_user

from config import (
    MODEL_PATH, LABELS_PATH, SCALER_PATH, MODELS_DIR,
    SECRET_KEY, DEBUG, HOST, PORT,
    ANGLE_TRIPLETS, MP_DETECTION_CONFIDENCE, IMAGE_SIZE,
    MIN_HOLD_SECONDS, MIN_POSE_SCORE, MIN_CONFIDENCE
)
from models          import db, bcrypt, User, PracticeSession, PoseLog, DailyStats
from auth            import auth
from dashboard       import dashboard_bp
from video_analyzer  import video_bp
from pose_corrector  import PoseCorrector, POSE_CATEGORY
from calorie_calculator import CalorieCalculator
from utils.label_mapper import get_label
from config import DATABASE_PATH

# ─── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DATABASE_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"]             = 500 * 1024 * 1024  # 500MB max upload

db.init_app(app)
bcrypt.init_app(app)

login_manager = LoginManager(app)
login_manager.login_view      = "auth.login"
login_manager.login_message   = "Please log in to access this page."

app.register_blueprint(auth)
app.register_blueprint(dashboard_bp)
app.register_blueprint(video_bp)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ─── Create tables ────────────────────────────────────────────────────────────
with app.app_context():
    db.create_all()
    print("Database tables created.")

# ─── Load ML model ────────────────────────────────────────────────────────────
print("Loading model...")
model         = joblib.load(MODEL_PATH)
scaler        = joblib.load(SCALER_PATH)
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

with open(LABELS_PATH) as f:
    raw_labels   = json.load(f)
class_labels = sorted(set(get_label(l) for l in raw_labels))
print(f"Model loaded — {len(class_labels)} classes")

# ─── MediaPipe ────────────────────────────────────────────────────────────────
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

pose_detector = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,
    smooth_landmarks=True,
    min_detection_confidence=MP_DETECTION_CONFIDENCE,
    min_tracking_confidence=0.5
)

corrector  = PoseCorrector()

# Per-user calorie calculators
user_calculators = {}

def get_calculator(user_id, weight_kg=65):
    if user_id not in user_calculators:
        user_calculators[user_id] = CalorieCalculator(weight_kg)
    return user_calculators[user_id]

# ─── Helpers ──────────────────────────────────────────────────────────────────
def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize(i) for i in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

def extract_features_from_landmarks(landmarks):
    coords = [v for lm in landmarks for v in [lm.x, lm.y, lm.z]]
    angles = [calculate_angle(
        (landmarks[a].x, landmarks[a].y),
        (landmarks[b].x, landmarks[b].y),
        (landmarks[c].x, landmarks[c].y)
    ) for a, b, c in ANGLE_TRIPLETS]
    vis = [lm.visibility for lm in landmarks]
    return np.array(coords + angles + vis, dtype=np.float32)

def get_joint_angles(landmarks):
    names = [
        "left_elbow_angle","right_elbow_angle",
        "left_shoulder_angle","right_shoulder_angle",
        "left_hip_angle","right_hip_angle",
        "left_knee_angle","right_knee_angle",
        "left_ankle_angle","right_ankle_angle",
        "shoulder_width_angle","hip_width_angle",
    ]
    return {
        name: calculate_angle(
            (landmarks[a].x, landmarks[a].y),
            (landmarks[b].x, landmarks[b].y),
            (landmarks[c].x, landmarks[c].y)
        )
        for name, (a, b, c) in zip(names, ANGLE_TRIPLETS)
    }

def update_daily_stats(user_id, calories, poses, time_sec):
    today = datetime.utcnow().date()
    stats = DailyStats.query.filter_by(user_id=user_id, date=today).first()
    if not stats:
        stats = DailyStats(user_id=user_id, date=today)
        db.session.add(stats)
    stats.total_calories += calories
    stats.total_poses    += poses
    stats.total_time_sec += time_sec
    stats.sessions_count += 1
    db.session.commit()

# ─── Active sessions per user ─────────────────────────────────────────────────
active_sessions = {}   # user_id → PracticeSession.id

# ─── Routes ───────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
        logged_in=current_user.is_authenticated,
        username=current_user.username if current_user.is_authenticated else ""
    )

@app.route("/practice")
@login_required
def practice():
    # Start a new DB session for this practice
    sess = PracticeSession(user_id=current_user.id, session_type="live")
    db.session.add(sess)
    db.session.commit()
    active_sessions[current_user.id] = sess.id

    calc = get_calculator(current_user.id, current_user.weight_kg)
    calc.reset()

    return render_template("practice.html", classes=class_labels)

@app.route("/session_summary")
@login_required
def session_summary():
    calc    = get_calculator(current_user.id)
    summary = calc.get_session_summary()
    return render_template("summary.html", summary=summary)

# ─── API: process frame ────────────────────────────────────────────────────────
@app.route("/api/process_frame", methods=["POST"])
@login_required
def process_frame():
    data        = request.get_json()
    image_data  = data.get("image", "")
    weight_kg   = float(data.get("weight", current_user.weight_kg))
    target_pose = get_label(data.get("target_pose", "").strip())

    calc             = get_calculator(current_user.id, weight_kg)
    calc.weight_kg   = weight_kg

    try:
        img_bytes = base64.b64decode(image_data.split(",")[1])
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if frame is None:
        return jsonify({"error": "Empty frame"}), 400

    frame_rgb = cv2.cvtColor(cv2.resize(frame, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
    results   = pose_detector.process(frame_rgb)

    if not results.pose_landmarks:
        return jsonify({"pose_detected": False,
                        "message": "No person detected"})

    landmarks       = results.pose_landmarks.landmark
    features        = extract_features_from_landmarks(landmarks)
    features_scaled = scaler.transform(features.reshape(1, -1))
    joint_angles    = get_joint_angles(landmarks)

    if hasattr(model, "predict_proba"):
        proba          = model.predict_proba(features_scaled)[0]
        top3_idx       = np.argsort(proba)[::-1][:3]
        top3_preds     = [{"pose": label_encoder.classes_[i],
                           "confidence": round(float(proba[i]) * 100, 1)}
                          for i in top3_idx]
        predicted_pose = top3_preds[0]["pose"]
        confidence     = top3_preds[0]["confidence"]
    else:
        predicted_pose = model.predict(features_scaled)[0]
        confidence     = 85.0
        top3_preds     = [{"pose": predicted_pose, "confidence": confidence}]

    if target_pose:
        result = corrector.is_target_pose(target_pose, joint_angles,
                                          predicted_pose, confidence)
        can_track = bool(result["is_performing"] and
                        len(result["corrections"]) == 0)
        response_data = {
            "pose_detected":   True,
            "mode":            "target",
            "target_pose":     target_pose,
            "predicted_pose":  predicted_pose,
            "predicted_match": bool(result["predicted_match"]),
            "is_performing":   bool(result["is_performing"]),
            "confidence":      confidence,
            "top3":            top3_preds,
            "score":           result["match_score"],
            "is_correct":      can_track,
            "corrections":     result["corrections"],
            "joint_status":    result["joint_status"],
            "has_reference":   result["has_reference"],
            "feedback":        result["feedback"],
            "can_track":       can_track,
            "joint_angles":    {k: round(v, 1) for k, v in joint_angles.items()},
            "calories_total":  round(calc.total_calories, 3),
        }
    else:
        correction = corrector.check_pose(predicted_pose, joint_angles)
        can_track  = (confidence >= MIN_CONFIDENCE * 100 and
                      correction["score"] >= MIN_POSE_SCORE)
        response_data = {
            "pose_detected":  True,
            "mode":           "free",
            "predicted_pose": predicted_pose,
            "confidence":     confidence,
            "top3":           top3_preds,
            "score":          correction["score"],
            "is_correct":     bool(correction["is_correct"]),
            "corrections":    correction["corrections"],
            "joint_status":   correction["joint_status"],
            "has_reference":  bool(correction["has_reference"]),
            "can_track":      bool(can_track),
            "feedback":       "",
            "joint_angles":   {k: round(v, 1) for k, v in joint_angles.items()},
            "calories_total": round(calc.total_calories, 3),
        }

    # REPLACE with this — draw landmarks THEN encode the drawn frame
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    mp_drawing.draw_landmarks(
        frame_bgr,                          # ← draw ON this variable
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
    )

    frame_small = cv2.resize(frame_bgr, (320, 240))   # ← encode the DRAWN frame
    _, buffer   = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 70])
    response_data["frame"] = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"

    
    return jsonify(sanitize(response_data)) 

# ─── API: log pose ─────────────────────────────────────────────────────────────
@app.route("/api/log_pose", methods=["POST"])
@login_required
def log_pose():
    data        = request.get_json()
    pose_name   = data.get("pose_name", "")
    duration    = float(data.get("duration_sec", 0))
    is_correct  = bool(data.get("is_correct", False))
    weight_kg   = float(data.get("weight", current_user.weight_kg))

    category = POSE_CATEGORY.get(pose_name.lower(), "default")
    from config import MET_VALUES
    met      = MET_VALUES.get(category, MET_VALUES["default"])
    calories = met * weight_kg * (duration / 3600) if is_correct else 0.0

    calc = get_calculator(current_user.id, weight_kg)
    calc.total_calories += calories
    calc.session_log.append({
        "pose": pose_name, "duration_sec": round(duration, 1),
        "calories": round(calories, 3), "is_correct": is_correct,
        "timestamp": time.strftime("%H:%M:%S"),
    })

    # Save to database
    session_id = active_sessions.get(current_user.id)
    if session_id:
        sess = PracticeSession.query.get(session_id)
        if sess:
            sess.total_calories += calories
            sess.total_poses    += 1
            sess.correct_poses  += (1 if is_correct else 0)
            sess.total_time_sec += duration
            sess.ended_at        = datetime.utcnow()

        db.session.add(PoseLog(
            session_id   = session_id,
            pose_name    = pose_name,
            duration_sec = round(duration, 1),
            score        = float(data.get("score", 0)),
            is_correct   = is_correct,
            calories     = round(calories, 3),
        ))
        update_daily_stats(current_user.id, calories, 1, duration)

    db.session.commit()

    return jsonify({
        "logged":         True,
        "calories_this":  round(calories, 3),
        "calories_total": round(calc.total_calories, 3),
        "session_log":    calc.session_log,
    })

# ─── API: reset session ────────────────────────────────────────────────────────
@app.route("/api/reset_session", methods=["POST"])
@login_required
def reset_session():
    calc = get_calculator(current_user.id)
    calc.reset()

    sess = PracticeSession(user_id=current_user.id, session_type="live")
    db.session.add(sess)
    db.session.commit()
    active_sessions[current_user.id] = sess.id

    return jsonify({"reset": True})

# ─── API: reference image ──────────────────────────────────────────────────────
@app.route("/api/reference_image", methods=["GET"])
def reference_image():
    from utils.label_mapper import get_label as gl
    pose_name = request.args.get("pose", "").strip()
    clean     = gl(pose_name)
    fname     = clean.replace(" ", "_") + ".jpg"
    img_path  = os.path.join(app.static_folder, "reference_poses", fname)
    if os.path.exists(img_path):
        return jsonify({"exists": True, "url": f"/static/reference_poses/{fname}"})
    fname2    = pose_name.replace(" ", "_") + ".jpg"
    img_path2 = os.path.join(app.static_folder, "reference_poses", fname2)
    if os.path.exists(img_path2):
        return jsonify({"exists": True, "url": f"/static/reference_poses/{fname2}"})
    return jsonify({"exists": False, "url": ""})

# ─── API: session summary ──────────────────────────────────────────────────────
@app.route("/api/session_summary", methods=["GET"])
@login_required
def get_summary():
    calc = get_calculator(current_user.id)
    return jsonify(calc.get_session_summary())

if __name__ == "__main__":
    app.run(debug=DEBUG, host=HOST, port=PORT)