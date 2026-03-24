# video_analyzer.py

import os
import cv2
import numpy as np
import joblib
import mediapipe as mp
from flask        import Blueprint, render_template, request, jsonify
from flask_login  import login_required, current_user
from werkzeug.utils import secure_filename
from models       import db, PracticeSession, PoseLog, DailyStats
from datetime     import datetime
from config       import (MODELS_DIR, STATIC_DIR, ANGLE_TRIPLETS,
                          IMAGE_SIZE, MP_DETECTION_CONFIDENCE,
                          MET_VALUES)
from pose_corrector import PoseCorrector, POSE_CATEGORY

video_bp  = Blueprint("video", __name__)
corrector = PoseCorrector()

UPLOAD_FOLDER   = os.path.join(STATIC_DIR, "uploads")
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "webm"}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model         = joblib.load(os.path.join(MODELS_DIR, "yoga_pose_model.pkl"))
scaler        = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
mp_pose       = mp.solutions.pose


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc  = a - b, c - b
    cosine  = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def extract_features(landmarks):
    coords = [v for lm in landmarks for v in [lm.x, lm.y, lm.z]]
    angles = [calculate_angle(
        (landmarks[a].x, landmarks[a].y),
        (landmarks[b].x, landmarks[b].y),
        (landmarks[c].x, landmarks[c].y)
    ) for a, b, c in ANGLE_TRIPLETS]
    vis = [lm.visibility for lm in landmarks]
    return np.array(coords + angles + vis, dtype=np.float32)


@video_bp.route("/video_upload")
@login_required
def video_upload_page():
    return render_template("video_upload.html")


@video_bp.route("/api/analyze_video", methods=["POST"])
@login_required
def analyze_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if not file or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Use mp4, mov, avi or webm"}), 400

    # Save uploaded file
    filename  = secure_filename(f"user{current_user.id}_{int(datetime.utcnow().timestamp())}.mp4")
    filepath  = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Analyze video
    results   = analyze_video_file(filepath, current_user.weight_kg)

    # Save session to database
    session_obj = PracticeSession(
        user_id        = current_user.id,
        started_at     = datetime.utcnow(),
        ended_at       = datetime.utcnow(),
        total_calories = results["total_calories"],
        total_poses    = results["total_poses"],
        correct_poses  = results["correct_poses"],
        total_time_sec = results["duration_sec"],
        session_type   = "video",
    )
    db.session.add(session_obj)
    db.session.flush()

    for log in results["pose_logs"]:
        db.session.add(PoseLog(
            session_id   = session_obj.id,
            pose_name    = log["pose"],
            duration_sec = log["duration_sec"],
            score        = log["score"],
            is_correct   = log["is_correct"],
            calories     = log["calories"],
        ))

    update_daily_stats(current_user.id, results)
    db.session.commit()

    # Clean up uploaded file
    os.remove(filepath)

    return jsonify({**results, "session_id": session_obj.id})


def analyze_video_file(filepath, weight_kg):
    cap         = cv2.VideoCapture(filepath)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec= total_frames / fps

    pose_detections = []
    frame_idx       = 0
    sample_every    = max(1, int(fps // 2))  # sample 2 frames per second

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=MP_DETECTION_CONFIDENCE
    ) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_every == 0:
                img_rgb  = cv2.cvtColor(cv2.resize(frame, IMAGE_SIZE), cv2.COLOR_BGR2RGB)
                results  = pose.process(img_rgb)

                if results.pose_landmarks:
                    lms      = results.pose_landmarks.landmark
                    features = extract_features(lms)
                    features_scaled = scaler.transform(features.reshape(1, -1))

                    proba          = model.predict_proba(features_scaled)[0]
                    top_idx        = np.argmax(proba)
                    predicted_pose = label_encoder.classes_[top_idx]
                    confidence     = float(proba[top_idx]) * 100

                    angle_names = [
                        "left_elbow_angle","right_elbow_angle",
                        "left_shoulder_angle","right_shoulder_angle",
                        "left_hip_angle","right_hip_angle",
                        "left_knee_angle","right_knee_angle",
                        "left_ankle_angle","right_ankle_angle",
                        "shoulder_width_angle","hip_width_angle",
                    ]
                    joint_angles = {}
                    for name, (a, b, c) in zip(angle_names, ANGLE_TRIPLETS):
                        joint_angles[name] = calculate_angle(
                            (lms[a].x, lms[a].y),
                            (lms[b].x, lms[b].y),
                            (lms[c].x, lms[c].y)
                        )

                    correction = corrector.check_pose(predicted_pose, joint_angles)
                    pose_detections.append({
                        "frame":          frame_idx,
                        "time_sec":       round(frame_idx / fps, 1),
                        "pose":           predicted_pose,
                        "confidence":     round(confidence, 1),
                        "score":          correction["score"],
                        "is_correct":     correction["is_correct"],
                        "corrections":    correction["corrections"],
                    })

            frame_idx += 1

    cap.release()

    # Group consecutive detections of same pose
    pose_logs      = group_pose_detections(pose_detections, sample_every, fps, weight_kg)
    total_calories = sum(p["calories"] for p in pose_logs)
    correct_poses  = sum(1 for p in pose_logs if p["is_correct"])

    return {
        "total_calories": round(total_calories, 3),
        "total_poses":    len(pose_logs),
        "correct_poses":  correct_poses,
        "duration_sec":   round(duration_sec, 1),
        "pose_logs":      pose_logs,
        "frame_count":    total_frames,
    }


def group_pose_detections(detections, sample_every, fps, weight_kg):
    """Group consecutive same-pose detections into pose events."""
    if not detections:
        return []

    logs         = []
    current_pose = detections[0]["pose"]
    start_time   = detections[0]["time_sec"]
    scores       = [detections[0]["score"]]

    for det in detections[1:]:
        if det["pose"] == current_pose:
            scores.append(det["score"])
        else:
            duration    = det["time_sec"] - start_time
            avg_score   = float(np.mean(scores))
            is_correct  = avg_score >= 60 and duration >= 2.0
            category    = POSE_CATEGORY.get(current_pose, "default")
            met         = MET_VALUES.get(category, MET_VALUES["default"])
            calories    = met * weight_kg * (duration / 3600) if is_correct else 0.0

            logs.append({
                "pose":        current_pose,
                "duration_sec": round(duration, 1),
                "score":       round(avg_score, 1),
                "is_correct":  is_correct,
                "calories":    round(calories, 4),
                "start_time":  round(start_time, 1),
            })

            current_pose = det["pose"]
            start_time   = det["time_sec"]
            scores       = [det["score"]]

    return logs


def update_daily_stats(user_id, results):
    today = datetime.utcnow().date()
    stats = DailyStats.query.filter_by(user_id=user_id, date=today).first()
    if not stats:
        stats = DailyStats(user_id=user_id, date=today)
        db.session.add(stats)
    stats.total_calories += results["total_calories"]
    stats.total_poses    += results["total_poses"]
    stats.total_time_sec += results["duration_sec"]
    stats.sessions_count += 1