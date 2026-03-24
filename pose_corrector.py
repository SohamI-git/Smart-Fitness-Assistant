import numpy as np
from config import ANGLE_THRESHOLD

POSE_REFERENCE = {
    "adho mukha svanasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (60,  15),
        "right_hip_angle":      (60,  15),
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
    },
    "adho mukha vrksasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
    },
    "akarna dhanurasana": {
        "left_knee_angle":      (170, 15),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (90,  15),
        "right_elbow_angle":    (60,  20),
    },
    "ananda balasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_hip_angle":       (90,  15),
        "right_hip_angle":      (90,  15),
    },
    "anantasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (90,  15),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (170, 15),
    },
    "anjaneyasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
    },
    "ardha chandrasana": {
        "left_knee_angle":      (175, 10),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (90,  15),
        "right_shoulder_angle": (90,  15),
    },
    "ardha matsyendrasana": {
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (120, 20),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (30,  20),
    },
    "ardha pincha mayurasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (60,  15),
    },
    "astavakrasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (90,  20),
        "right_knee_angle":     (90,  20),
    },
    "baddha konasana": {
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (70,  15),
        "right_hip_angle":      (70,  15),
        "left_shoulder_angle":  (20,  20),
    },
    "bakasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
    },
    "balasana": {
        "left_knee_angle":      (30,  15),
        "right_knee_angle":     (30,  15),
        "left_hip_angle":       (40,  15),
        "right_hip_angle":      (40,  15),
        "left_shoulder_angle":  (170, 15),
    },
    "bharadvajasana i": {
        "left_knee_angle":      (90,  20),
        "right_knee_angle":     (90,  20),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (45,  20),
        "right_shoulder_angle": (30,  20),
    },
    "bhekasana": {
        "left_knee_angle":      (30,  20),
        "right_knee_angle":     (30,  20),
        "left_shoulder_angle":  (45,  20),
        "right_shoulder_angle": (45,  20),
    },
    "bhujangasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_shoulder_angle":  (45,  15),
        "right_shoulder_angle": (45,  15),
        "left_hip_angle":       (170, 10),
    },
    "bhujapidasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (90,  20),
        "right_knee_angle":     (90,  20),
    },
    "camatkarasana": {
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (120, 15),
    },
    "chaturanga dandasana": {
        "left_elbow_angle":     (90,  10),
        "right_elbow_angle":    (90,  10),
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (175, 10),
    },
    "dandasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (90,  15),
        "right_hip_angle":      (90,  15),
        "left_shoulder_angle":  (20,  15),
    },
    "dhanurasana": {
        "left_knee_angle":      (45,  15),
        "right_knee_angle":     (45,  15),
        "left_shoulder_angle":  (60,  20),
        "right_shoulder_angle": (60,  20),
        "left_hip_angle":       (150, 15),
    },
    "dwi pada viparita dandasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_hip_angle":       (130, 15),
    },
    "eka pada koundinyanasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (170, 15),
        "right_knee_angle":     (90,  20),
    },
    "garudasana": {
        "left_knee_angle":      (120, 20),
        "right_knee_angle":     (120, 20),
        "left_elbow_angle":     (90,  20),
        "right_elbow_angle":    (90,  20),
        "left_hip_angle":       (120, 15),
    },
    "gomukhasana": {
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (80,  15),
        "left_elbow_angle":     (60,  20),
        "right_elbow_angle":    (60,  20),
    },
    "halasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (30,  20),
        "right_hip_angle":      (30,  20),
        "left_shoulder_angle":  (30,  20),
    },
    "hanumanasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (170, 10),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
    },
    "janu sirsasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (50,  15),
        "right_hip_angle":      (90,  15),
    },
    "kapotasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_shoulder_angle":  (150, 20),
        "right_shoulder_angle": (150, 20),
        "left_hip_angle":       (90,  15),
    },
    "krounchasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (60,  15),
        "left_shoulder_angle":  (170, 15),
    },
    "kurmasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (40,  15),
        "right_hip_angle":      (40,  15),
        "left_elbow_angle":     (170, 10),
    },
    "lolasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (60,  20),
    },
    "makara adho mukha svanasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (175, 10),
    },
    "malasana": {
        "left_knee_angle":      (60,  15),
        "right_knee_angle":     (60,  15),
        "left_hip_angle":       (60,  20),
        "right_hip_angle":      (60,  20),
        "left_elbow_angle":     (90,  20),
    },
    "marjaryasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
        "left_hip_angle":       (90,  15),
    },
    "matsyasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_shoulder_angle":  (150, 20),
    },
    "mayurasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (175, 10),
    },
    "natarajasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (60,  15),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (60,  20),
        "left_hip_angle":       (175, 10),
    },
    "padmasana": {
        "left_knee_angle":      (40,  15),
        "right_knee_angle":     (40,  15),
        "left_hip_angle":       (70,  15),
        "right_hip_angle":      (70,  15),
        "left_shoulder_angle":  (20,  20),
    },
    "parighasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (90,  15),
        "right_shoulder_angle": (170, 15),
    },
    "paripurna navasana": {
        "left_knee_angle":      (150, 15),
        "right_knee_angle":     (150, 15),
        "left_hip_angle":       (80,  15),
        "right_hip_angle":      (80,  15),
        "left_shoulder_angle":  (90,  15),
    },
    "parivrtta janu sirsasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (50,  15),
        "left_shoulder_angle":  (150, 20),
    },
    "parsva bakasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (60,  20),
    },
    "parsvottanasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (60,  15),
        "right_hip_angle":      (90,  15),
        "left_shoulder_angle":  (20,  20),
    },
    "pasasana": {
        "left_knee_angle":      (60,  15),
        "right_knee_angle":     (60,  15),
        "left_hip_angle":       (60,  15),
        "left_elbow_angle":     (90,  20),
        "right_shoulder_angle": (60,  20),
    },
    "paschimottanasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (50,  15),
        "right_hip_angle":      (50,  15),
        "left_shoulder_angle":  (150, 20),
    },
    "pawanmuktasana": {
        "left_knee_angle":      (60,  15),
        "right_knee_angle":     (60,  15),
        "left_hip_angle":       (60,  15),
        "right_hip_angle":      (60,  15),
        "left_shoulder_angle":  (30,  20),
    },
    "phalakasana": {
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (175, 10),
    },
    "pincha mayurasana": {
        "left_elbow_angle":     (90,  10),
        "right_elbow_angle":    (90,  10),
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_shoulder_angle":  (170, 15),
    },
    "prasarita padottanasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (50,  15),
        "right_hip_angle":      (50,  15),
        "left_shoulder_angle":  (90,  15),
    },
    "purvottanasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (150, 15),
        "right_hip_angle":      (150, 15),
        "left_elbow_angle":     (170, 10),
    },
    "rajakapotasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (150, 20),
        "right_shoulder_angle": (150, 20),
    },
    "salabhasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_shoulder_angle":  (20,  20),
        "right_shoulder_angle": (20,  20),
        "left_hip_angle":       (170, 10),
    },
    "salamba sarvangasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (175, 10),
        "right_hip_angle":      (175, 10),
        "left_elbow_angle":     (90,  15),
    },
    "salamba sirsasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (175, 10),
        "right_hip_angle":      (175, 10),
        "left_elbow_angle":     (90,  15),
    },
    "savasana": {
        "left_knee_angle":      (170, 15),
        "right_knee_angle":     (170, 15),
        "left_hip_angle":       (170, 15),
        "right_hip_angle":      (170, 15),
        "left_elbow_angle":     (160, 20),
        "right_elbow_angle":    (160, 20),
    },
    "setu bandha sarvangasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_hip_angle":       (130, 15),
        "right_hip_angle":      (130, 15),
        "left_shoulder_angle":  (30,  20),
    },
    "sukhasana": {
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (80,  15),
        "right_hip_angle":      (80,  15),
        "left_shoulder_angle":  (20,  20),
    },
    "supta baddha konasana": {
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (90,  15),
        "right_hip_angle":      (90,  15),
        "left_shoulder_angle":  (90,  20),
    },
    "supta padangusthasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (90,  15),
        "right_hip_angle":      (170, 10),
    },
    "supta virasana": {
        "left_knee_angle":      (30,  20),
        "right_knee_angle":     (30,  20),
        "left_hip_angle":       (170, 10),
        "right_hip_angle":      (170, 10),
        "left_shoulder_angle":  (170, 15),
    },
    "tadasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (175, 10),
        "right_hip_angle":      (175, 10),
        "left_shoulder_angle":  (10,  15),
        "right_shoulder_angle": (10,  15),
    },
    "tittibhasana": {
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
        "left_knee_angle":      (130, 20),
        "right_knee_angle":     (130, 20),
        "left_hip_angle":       (80,  15),
    },
    "tolasana": {
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
        "left_knee_angle":      (40,  20),
        "right_knee_angle":     (40,  20),
        "left_hip_angle":       (70,  15),
    },
    "upavistha konasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (60,  15),
        "right_hip_angle":      (60,  15),
        "left_shoulder_angle":  (150, 20),
    },
    "urdhva dhanurasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_hip_angle":       (130, 15),
        "right_hip_angle":      (130, 15),
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
    },
    "urdhva prasarita eka padasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
    },
    "ustrasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_shoulder_angle":  (90,  20),
        "right_shoulder_angle": (90,  20),
        "left_hip_angle":       (150, 15),
    },
    "utkatasana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_hip_angle":       (90,  15),
        "right_hip_angle":      (90,  15),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
    },
    "uttana shishosana": {
        "left_knee_angle":      (90,  15),
        "right_knee_angle":     (90,  15),
        "left_elbow_angle":     (170, 10),
        "right_elbow_angle":    (170, 10),
        "left_hip_angle":       (90,  15),
    },
    "uttanasana": {
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (45,  15),
        "right_hip_angle":      (45,  15),
        "left_shoulder_angle":  (150, 20),
    },
    "utthita hasta padangustasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (150, 15),
        "left_hip_angle":       (175, 10),
        "right_hip_angle":      (90,  15),
        "left_shoulder_angle":  (90,  15),
    },
    "utthita parsvakonasana": {
        "left_knee_angle":      (90,  10),
        "right_knee_angle":     (175, 10),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (90,  15),
        "left_hip_angle":       (90,  15),
    },
    "utthita trikonasana": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_shoulder_angle":  (90,  15),
        "right_shoulder_angle": (90,  15),
        "left_hip_angle":       (90,  15),
    },
    "vajrasana": {
        "left_knee_angle":      (30,  15),
        "right_knee_angle":     (30,  15),
        "left_hip_angle":       (150, 15),
        "right_hip_angle":      (150, 15),
        "left_shoulder_angle":  (20,  20),
    },
    "vasisthasana": {
        "left_elbow_angle":     (170, 10),
        "left_knee_angle":      (170, 10),
        "right_knee_angle":     (170, 10),
        "left_shoulder_angle":  (90,  15),
        "left_hip_angle":       (170, 10),
    },
    "viparita karani": {
        "left_knee_angle":      (175, 10),
        "right_knee_angle":     (175, 10),
        "left_hip_angle":       (90,  15),
        "right_hip_angle":      (90,  15),
        "left_shoulder_angle":  (30,  20),
    },
    "viparita virabhadrasana": {
        "left_knee_angle":      (90,  10),
        "right_knee_angle":     (175, 10),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (90,  15),
        "left_hip_angle":       (90,  15),
    },
    "virabhadrasana i": {
        "left_knee_angle":      (90,  10),
        "right_knee_angle":     (170, 10),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
    },
    "virabhadrasana ii": {
        "left_knee_angle":      (90,  10),
        "right_knee_angle":     (170, 10),
        "left_shoulder_angle":  (90,  15),
        "right_shoulder_angle": (90,  15),
        "left_hip_angle":       (90,  15),
    },
    "virabhadrasana iii": {
        "right_knee_angle":     (170, 10),
        "left_knee_angle":      (170, 10),
        "left_hip_angle":       (90,  15),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
    },
    "virasana": {
        "left_knee_angle":      (30,  15),
        "right_knee_angle":     (30,  15),
        "left_hip_angle":       (150, 15),
        "right_hip_angle":      (150, 15),
        "left_shoulder_angle":  (20,  20),
    },
    "vrischikasana": {
        "left_elbow_angle":     (90,  15),
        "right_elbow_angle":    (90,  15),
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (120, 20),
    },
    "vriksasana": {
        "left_knee_angle":      (175, 10),
        "left_shoulder_angle":  (170, 15),
        "right_shoulder_angle": (170, 15),
        "left_hip_angle":       (175, 10),
        "right_hip_angle":      (60,  20),
    },
    "yoganidrasana": {
        "left_knee_angle":      (60,  20),
        "right_knee_angle":     (60,  20),
        "left_hip_angle":       (60,  15),
        "right_hip_angle":      (60,  15),
        "left_shoulder_angle":  (90,  20),
    },
}

CORRECTION_MESSAGES = {
    "left_knee_angle":      {"too_low": "Bend your left knee more",       "too_high": "Straighten your left knee"},
    "right_knee_angle":     {"too_low": "Bend your right knee more",      "too_high": "Straighten your right knee"},
    "left_hip_angle":       {"too_low": "Lower your hips / bend forward", "too_high": "Open up your left hip more"},
    "right_hip_angle":      {"too_low": "Lower your hips / bend forward", "too_high": "Open up your right hip more"},
    "left_elbow_angle":     {"too_low": "Straighten your left elbow",     "too_high": "Bend your left elbow more"},
    "right_elbow_angle":    {"too_low": "Straighten your right elbow",    "too_high": "Bend your right elbow more"},
    "left_shoulder_angle":  {"too_low": "Raise your left arm higher",     "too_high": "Lower your left arm"},
    "right_shoulder_angle": {"too_low": "Raise your right arm higher",    "too_high": "Lower your right arm"},
    "left_ankle_angle":     {"too_low": "Flex your left foot more",       "too_high": "Point your left foot less"},
    "right_ankle_angle":    {"too_low": "Flex your right foot more",      "too_high": "Point your right foot less"},
}

POSE_CATEGORY = {
    "tadasana": "standing",               "vriksasana": "balancing",
    "virabhadrasana i": "standing",       "virabhadrasana ii": "standing",
    "virabhadrasana iii": "balancing",    "viparita virabhadrasana": "standing",
    "uttanasana": "forward_bend",         "ardha uttanasana": "forward_bend",
    "utkatasana": "standing",             "adho mukha svanasana": "forward_bend",
    "adho mukha vrksasana": "inverted",   "bhujangasana": "prone",
    "balasana": "prone",                  "savasana": "supine",
    "paschimottanasana": "seated",        "dandasana": "seated",
    "padmasana": "seated",                "sukhasana": "seated",
    "dhanurasana": "prone",               "ustrasana": "backbend",
    "setu bandha sarvangasana": "backbend","natarajasana": "balancing",
    "bakasana": "balancing",              "vasisthasana": "balancing",
    "garudasana": "balancing",            "ardha chandrasana": "balancing",
    "gomukhasana": "seated",              "malasana": "seated",
    "utthita trikonasana": "standing",    "utthita parsvakonasana": "standing",
    "prasarita padottanasana": "standing","parsvottanasana": "standing",
    "parivrtta janu sirsasana": "seated", "janu sirsasana": "seated",
    "baddha konasana": "seated",          "upavistha konasana": "seated",
    "paripurna navasana": "seated",       "halasana": "inverted",
    "salamba sarvangasana": "inverted",   "salamba sirsasana": "inverted",
    "viparita karani": "inverted",        "urdhva dhanurasana": "backbend",
    "kapotasana": "backbend",             "rajakapotasana": "backbend",
    "salabhasana": "prone",               "makarasana": "prone",
    "chaturanga dandasana": "prone",      "phalakasana": "prone",
    "mayurasana": "balancing",            "pincha mayurasana": "inverted",
    "tittibhasana": "balancing",          "astavakrasana": "balancing",
    "bhujapidasana": "balancing",         "lolasana": "balancing",
    "tolasana": "balancing",              "hanumanasana": "seated",
    "anjaneyasana": "standing",           "ardha matsyendrasana": "twist",
    "bharadvajasana i": "twist",          "pasasana": "twist",
    "pawanmuktasana": "supine",           "supta baddha konasana": "supine",
    "supta padangusthasana": "supine",    "supta virasana": "supine",
    "ananda balasana": "supine",          "anantasana": "supine",
    "yoganidrasana": "supine",            "virasana": "seated",
    "vajrasana": "seated",                "kurmasana": "seated",
    "krounchasana": "seated",             "marjaryasana": "prone",
    "makara adho mukha svanasana": "prone","uttana shishosana": "prone",
    "camatkarasana": "backbend",          "dwi pada viparita dandasana": "backbend",
    "purvottanasana": "backbend",         "matsyasana": "supine",
    "urdhva prasarita eka padasana": "balancing",
    "utthita hasta padangustasana": "balancing",
    "ardha pincha mayurasana": "forward_bend",
    "eka pada koundinyanasana": "balancing",
    "parsva bakasana": "balancing",       "vrischikasana": "inverted",
    "parighasana": "standing",            "akarna dhanurasana": "seated",
    "bhekasana": "prone",                 "dhanurasana": "prone",
    "pincha mayurasana": "inverted",      "adho mukha vrksasana": "inverted",
}


class PoseCorrector:

    def __init__(self):
        self.reference = POSE_REFERENCE

    def check_pose(self, pose_name, current_angles: dict):
        pose_key = pose_name.lower().strip()

        if pose_key not in self.reference:
            return {
                "is_correct":    False,
                "score":         0.0,
                "corrections":   [f"No reference data for '{pose_name}' yet — keep practicing!"],
                "joint_status":  {},
                "has_reference": False,
            }

        ref_joints   = self.reference[pose_key]
        corrections  = []
        joint_status = {}
        angle_errors = []

        for joint_name, (ideal, tolerance) in ref_joints.items():
            current = current_angles.get(joint_name)
            if current is None:
                continue

            error = current - ideal
            angle_errors.append(abs(error))

            if abs(error) <= tolerance:
                joint_status[joint_name] = "correct"
            elif error < 0:
                joint_status[joint_name] = "too_low"
                msg = CORRECTION_MESSAGES.get(joint_name, {}).get(
                    "too_low", f"{joint_name} too low ({current:.0f}° vs ideal {ideal}°)")
                corrections.append(msg)
            else:
                joint_status[joint_name] = "too_high"
                msg = CORRECTION_MESSAGES.get(joint_name, {}).get(
                    "too_high", f"{joint_name} too high ({current:.0f}° vs ideal {ideal}°)")
                corrections.append(msg)

        score = max(0.0, 100.0 - (np.mean(angle_errors) * 1.5)) if angle_errors else 100.0

        return {
            "is_correct":    len(corrections) == 0,
            "score":         round(score, 1),
            "corrections":   corrections,
            "joint_status":  joint_status,
            "has_reference": True,
        }

    def get_pose_category(self, pose_name):
        return POSE_CATEGORY.get(pose_name.lower().strip(), "default")


    # Add this method inside the PoseCorrector class in pose_corrector.py

    def is_target_pose(self, target_pose, current_angles, predicted_pose, confidence):
        """
        When user has selected a target pose, check if they are
        actually performing it rather than guessing from all 82 poses.

        Returns a detailed result dict with:
        - is_performing : bool  — are they doing the right pose?
        - match_score   : float — how close their angles are to target (0-100)
        - corrections   : list  — what to fix
        - feedback      : str   — one-line human message
        """
        target_key = target_pose.lower().strip()

        # Step 1 — check if predicted pose matches target
        predicted_match = predicted_pose.lower().strip() == target_key

        # Step 2 — check angles against reference regardless of prediction
        angle_check = self.check_pose(target_pose, current_angles)

        # Step 3 — combine both signals
        # Pose is considered correct only if:
        #   a) model predicts it matches AND angle score is above threshold
        #   b) OR angle score is very high (>= 80) even if model is uncertain
        angle_score     = angle_check["score"]
        is_performing = bool((predicted_match and angle_score >= 60) or (angle_score >= 80))

        # Step 4 — build human-readable feedback
        if is_performing and len(angle_check["corrections"]) == 0:
            feedback = f"Perfect {target_pose}! Hold this pose."
        elif is_performing and len(angle_check["corrections"]) > 0:
            feedback = f"Good attempt at {target_pose}. Refine your form."
        elif not predicted_match and angle_score < 40:
            feedback = f"This does not look like {target_pose}. Check the reference image."
        else:
            feedback = f"Getting closer to {target_pose}. Follow the corrections below."

        return {
            "is_performing":   is_performing,
            "match_score":     angle_score,
            "corrections":     angle_check["corrections"],
            "joint_status":    angle_check["joint_status"],
            "has_reference":   angle_check["has_reference"],
            "predicted_pose":  predicted_pose,
            "predicted_match": bool(predicted_match),
            "confidence":      confidence,
            "feedback":        feedback,
        }