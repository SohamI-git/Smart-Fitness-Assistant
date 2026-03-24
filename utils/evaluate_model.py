# utils/evaluate_model.py

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score
)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATASET_DIR, MODELS_DIR, MODEL_PATH, SCALER_PATH

# ─── Load ─────────────────────────────────────────────────────────────────────
print("Loading model and test data...")
model   = joblib.load(MODEL_PATH)
scaler  = joblib.load(SCALER_PATH)
le      = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

df        = pd.read_csv(os.path.join(DATASET_DIR, "test_keypoints.csv"))
feat_cols = [c for c in df.columns if c not in ["label", "filename"]]
X_test    = scaler.transform(df[feat_cols].values.astype(np.float32))
y_true    = df["label"].values
y_pred    = model.predict(X_test)

classes   = le.classes_

# ─── Global metrics ───────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  GLOBAL METRICS")
print("="*60)
print(f"  Total test samples : {len(y_true)}")
print(f"  Total classes      : {len(classes)}")
print(f"  Overall accuracy   : {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"  Macro precision    : {precision_score(y_true, y_pred, average='macro',  zero_division=0)*100:.2f}%")
print(f"  Macro recall       : {recall_score(   y_true, y_pred, average='macro',  zero_division=0)*100:.2f}%")
print(f"  Macro F1-score     : {f1_score(       y_true, y_pred, average='macro',  zero_division=0)*100:.2f}%")
print(f"  Weighted F1-score  : {f1_score(       y_true, y_pred, average='weighted',zero_division=0)*100:.2f}%")

# ─── Per-class TP, TN, FP, FN ────────────────────────────────────────────────
print("\n" + "="*60)
print("  PER-CLASS TP / FP / FN / TN / PRECISION / RECALL / F1")
print("="*60)

cm = confusion_matrix(y_true, y_pred, labels=classes)

records = []
for i, cls in enumerate(classes):
    TP = cm[i, i]
    FP = cm[:, i].sum() - TP        # other classes predicted as this class
    FN = cm[i, :].sum() - TP        # this class predicted as other
    TN = cm.sum() - TP - FP - FN    # everything else correct

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    records.append({
        "class":       cls,
        "TP":          int(TP),
        "FP":          int(FP),
        "FN":          int(FN),
        "TN":          int(TN),
        "precision":   round(precision, 3),
        "recall":      round(recall, 3),
        "f1":          round(f1, 3),
        "specificity": round(specificity, 3),
    })

results_df = pd.DataFrame(records)
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 120)
print(results_df.to_string(index=False))

# ─── Save CSV ────────────────────────────────────────────────────────────────
out_csv = os.path.join(MODELS_DIR, "per_class_metrics.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nPer-class metrics saved → {out_csv}")

# ─── Best and worst classes ───────────────────────────────────────────────────
print("\n" + "="*60)
print("  TOP 10 BEST PERFORMING CLASSES (by F1)")
print("="*60)
print(results_df.nlargest(10, "f1")[["class","TP","FP","FN","f1","precision","recall"]].to_string(index=False))

print("\n" + "="*60)
print("  TOP 10 WORST PERFORMING CLASSES (by F1)")
print("="*60)
print(results_df.nsmallest(10, "f1")[["class","TP","FP","FN","f1","precision","recall"]].to_string(index=False))

# ─── Confusion matrix heatmap (top 20 classes) ───────────────────────────────
from collections import Counter
top20      = [c for c, _ in Counter(y_true).most_common(20)]
mask       = np.isin(y_true, top20)
cm20       = confusion_matrix(y_true[mask], y_pred[mask], labels=top20)
cm20_norm  = cm20.astype(float) / (cm20.sum(axis=1, keepdims=True) + 1e-8)

fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm20_norm, annot=True, fmt=".2f",
            xticklabels=top20, yticklabels=top20,
            cmap="Blues", ax=ax, vmin=0, vmax=1,
            linewidths=0.3)
ax.set_title("Confusion matrix — top 20 classes (row-normalized)\nDiagonal = correct predictions")
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(rotation=0, fontsize=8)
plt.tight_layout()
cm_path = os.path.join(MODELS_DIR, "confusion_matrix_top20.png")
plt.savefig(cm_path, dpi=150)
plt.close()
print(f"\nConfusion matrix saved → {cm_path}")

# ─── TP/FP/FN bar chart ──────────────────────────────────────────────────────
top15 = results_df.nlargest(15, "TP")
fig, ax = plt.subplots(figsize=(14, 6))
x   = np.arange(len(top15))
w   = 0.25
ax.bar(x - w, top15["TP"], w, label="True Positive",  color="#4caf8a")
ax.bar(x,     top15["FP"], w, label="False Positive", color="#e05252")
ax.bar(x + w, top15["FN"], w, label="False Negative", color="#f0a500")
ax.set_xticks(x)
ax.set_xticklabels(top15["class"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Count")
ax.set_title("TP / FP / FN — top 15 classes by TP count")
ax.legend()
plt.tight_layout()
bar_path = os.path.join(MODELS_DIR, "tp_fp_fn_chart.png")
plt.savefig(bar_path, dpi=150)
plt.close()
print(f"TP/FP/FN chart saved → {bar_path}")

# ─── Precision-Recall scatter ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(results_df["recall"], results_df["precision"],
           c=results_df["f1"], cmap="RdYlGn",
           s=60, alpha=0.8, edgecolors="none")
for _, row in results_df.iterrows():
    ax.annotate(row["class"], (row["recall"], row["precision"]),
                fontsize=5, alpha=0.7)
ax.set_xlabel("Recall (sensitivity)")
ax.set_ylabel("Precision")
ax.set_title("Precision vs Recall — all 82 classes\nColor = F1 score (green = high)")
ax.axhline(0.77, color="gray", linestyle="--", alpha=0.4, label="Mean accuracy line")
plt.colorbar(ax.collections[0], label="F1 score")
plt.tight_layout()
pr_path = os.path.join(MODELS_DIR, "precision_recall_scatter.png")
plt.savefig(pr_path, dpi=150)
plt.close()
print(f"Precision-Recall scatter saved → {pr_path}")

print("\nEvaluation complete!")
print(f"All outputs saved in: {MODELS_DIR}")