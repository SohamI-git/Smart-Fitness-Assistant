import time
from config import MET_VALUES, DEFAULT_WEIGHT_KG
from pose_corrector import PoseCorrector

corrector = PoseCorrector()

class CalorieCalculator:

    def __init__(self, weight_kg=DEFAULT_WEIGHT_KG):
        self.weight_kg      = weight_kg
        self.total_calories = 0.0
        self.session_log    = []
        self._pose_start    = None
        self._current_pose  = None

    def start_pose(self, pose_name):
        self._current_pose = pose_name
        self._pose_start   = time.time()

    def end_pose(self, pose_name, is_correct=True):
        if self._pose_start is None or self._current_pose != pose_name:
            return 0.0
        duration_sec = time.time() - self._pose_start
        duration_hrs = duration_sec / 3600.0
        calories = 0.0
        if is_correct and duration_sec >= 2.0:
            category = corrector.get_pose_category(pose_name)
            met      = MET_VALUES.get(category, MET_VALUES["default"])
            calories = met * self.weight_kg * duration_hrs
        self.total_calories += calories
        self.session_log.append({
            "pose":         pose_name,
            "duration_sec": round(duration_sec, 1),
            "calories":     round(calories, 3),
            "is_correct":   is_correct,
            "timestamp":    time.strftime("%H:%M:%S"),
        })
        self._pose_start   = None
        self._current_pose = None
        return round(calories, 3)

    def get_session_summary(self):
        total_time = sum(e["duration_sec"] for e in self.session_log)
        correct    = sum(1 for e in self.session_log if e["is_correct"])
        total      = len(self.session_log)
        return {
            "total_calories":  round(self.total_calories, 2),
            "total_poses":     total,
            "correct_poses":   correct,
            "accuracy_pct":    round(correct / total * 100, 1) if total > 0 else 0,
            "total_time_sec":  round(total_time, 1),
            "session_log":     self.session_log,
        }

    def reset(self):
        self.total_calories = 0.0
        self.session_log    = []
        self._pose_start    = None
        self._current_pose  = None
