# models.py

from flask_sqlalchemy import SQLAlchemy
from flask_login      import UserMixin
from flask_bcrypt     import Bcrypt
from datetime         import datetime

db     = SQLAlchemy()
bcrypt = Bcrypt()


class User(UserMixin, db.Model):
    __tablename__ = "users"
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80),  unique=True, nullable=False)
    email         = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    weight_kg     = db.Column(db.Float, default=65.0)
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)

    sessions      = db.relationship("PracticeSession", backref="user", lazy=True)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class PracticeSession(db.Model):
    __tablename__    = "practice_sessions"
    id               = db.Column(db.Integer, primary_key=True)
    user_id          = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    started_at       = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at         = db.Column(db.DateTime, nullable=True)
    total_calories   = db.Column(db.Float,   default=0.0)
    total_poses      = db.Column(db.Integer, default=0)
    correct_poses    = db.Column(db.Integer, default=0)
    total_time_sec   = db.Column(db.Float,   default=0.0)
    session_type     = db.Column(db.String(20), default="live")  # live / video

    pose_logs        = db.relationship("PoseLog", backref="session", lazy=True)

    @property
    def accuracy_pct(self):
        if self.total_poses == 0:
            return 0.0
        return round(self.correct_poses / self.total_poses * 100, 1)

    @property
    def duration_minutes(self):
        return round(self.total_time_sec / 60, 1)


class PoseLog(db.Model):
    __tablename__  = "pose_logs"
    id             = db.Column(db.Integer, primary_key=True)
    session_id     = db.Column(db.Integer, db.ForeignKey("practice_sessions.id"), nullable=False)
    pose_name      = db.Column(db.String(100), nullable=False)
    duration_sec   = db.Column(db.Float,   default=0.0)
    score          = db.Column(db.Float,   default=0.0)
    is_correct     = db.Column(db.Boolean, default=False)
    calories       = db.Column(db.Float,   default=0.0)
    logged_at      = db.Column(db.DateTime, default=datetime.utcnow)


class DailyStats(db.Model):
    __tablename__  = "daily_stats"
    id             = db.Column(db.Integer, primary_key=True)
    user_id        = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    date           = db.Column(db.Date,    nullable=False)
    total_calories = db.Column(db.Float,   default=0.0)
    total_poses    = db.Column(db.Integer, default=0)
    total_time_sec = db.Column(db.Float,   default=0.0)
    sessions_count = db.Column(db.Integer, default=0)
    __table_args__ = (db.UniqueConstraint("user_id", "date"),)