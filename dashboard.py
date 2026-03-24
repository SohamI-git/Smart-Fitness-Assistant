# dashboard.py

from flask       import Blueprint, render_template, jsonify
from flask_login import login_required, current_user
from models      import db, PracticeSession, PoseLog, DailyStats
from datetime    import datetime, timedelta
from sqlalchemy  import func

dashboard_bp = Blueprint("dashboard", __name__)

@dashboard_bp.route("/dashboard")
@login_required
def dashboard():
    today      = datetime.utcnow().date()
    now        = datetime.utcnow()

    # Greeting based on time of day
    hour = now.hour
    if hour < 12:
        time_greeting = "morning"
    elif hour < 17:
        time_greeting = "afternoon"
    else:
        time_greeting = "evening"

    today_str = today.strftime("%A, %d %B %Y")

    # Today's stats from DailyStats
    today_stats = DailyStats.query.filter_by(
        user_id=current_user.id, date=today
    ).first() or DailyStats(
        total_calories=0, total_poses=0,
        total_time_sec=0, sessions_count=0
    )

    # Today's pose logs — all PoseLogs from today's sessions
    today_logs = db.session.query(PoseLog).join(PracticeSession).filter(
        PracticeSession.user_id == current_user.id,
        PoseLog.logged_at >= datetime.combine(today, datetime.min.time())
    ).order_by(PoseLog.logged_at.desc()).limit(50).all()

    # Today's accuracy
    today_correct  = sum(1 for l in today_logs if l.is_correct)
    total_today    = len(today_logs)
    today_accuracy = round(today_correct / total_today * 100) if total_today > 0 else 0

    # Daily calorie goal (simple formula: 5 kcal per kg bodyweight)
    daily_goal = round(current_user.weight_kg * 0.08, 2)
    goal_pct   = round((today_stats.total_calories or 0) / daily_goal * 100) \
                 if daily_goal > 0 else 0

    # All-time totals
    totals = db.session.query(
        func.sum(PracticeSession.total_calories).label("calories"),
        func.sum(PracticeSession.total_poses).label("poses"),
        func.count(PracticeSession.id).label("sessions"),
        func.sum(PracticeSession.total_time_sec).label("time_sec"),
    ).filter(PracticeSession.user_id == current_user.id).first()

    # Top poses with best score
    top_poses = db.session.query(
        PoseLog.pose_name,
        func.count(PoseLog.id).label("count"),
        func.avg(PoseLog.score).label("avg_score"),
        func.max(PoseLog.score).label("best_score"),
    ).join(PracticeSession).filter(
        PracticeSession.user_id == current_user.id
    ).group_by(PoseLog.pose_name).order_by(
        func.count(PoseLog.id).desc()
    ).limit(10).all()

    streak = calculate_streak(current_user.id)

    return render_template("dashboard.html",
        time_greeting  = time_greeting,
        today_str      = today_str,
        today_stats    = today_stats,
        today_logs     = today_logs,
        today_correct  = today_correct,
        today_accuracy = today_accuracy,
        daily_goal     = daily_goal,
        goal_pct       = goal_pct,
        totals         = totals,
        top_poses      = top_poses,
        streak         = streak,
        today          = today,
    )


@dashboard_bp.route("/api/dashboard_data")
@login_required
def dashboard_data():
    today      = datetime.utcnow().date()
    week_start = today - timedelta(days=6)

    daily = DailyStats.query.filter(
        DailyStats.user_id == current_user.id,
        DailyStats.date    >= week_start
    ).order_by(DailyStats.date).all()

    date_map                        = {d.date: d for d in daily}
    labels, calories, poses, minutes = [], [], [], []

    for i in range(7):
        day = week_start + timedelta(days=i)
        labels.append(day.strftime("%a %d"))
        if day in date_map:
            d = date_map[day]
            calories.append(round(d.total_calories, 2))
            poses.append(d.total_poses)
            minutes.append(round(d.total_time_sec / 60, 1))
        else:
            calories.append(0)
            poses.append(0)
            minutes.append(0)

    return jsonify({
        "labels":   labels,
        "calories": calories,
        "poses":    poses,
        "minutes":  minutes,
    })


def calculate_streak(user_id):
    today  = datetime.utcnow().date()
    streak = 0
    day    = today
    while True:
        exists = DailyStats.query.filter_by(
            user_id=user_id, date=day
        ).filter(DailyStats.sessions_count > 0).first()
        if exists:
            streak += 1
            day -= timedelta(days=1)
        else:
            break
    return streak