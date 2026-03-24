# train_model.py  (final version with augmentation + XGBoost)

import os, json, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm             import SVC
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.metrics         import accuracy_score, classification_report, top_k_accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost                 import XGBClassifier

from config import DATASET_DIR, MODELS_DIR, MODEL_PATH, LABELS_PATH, SCALER_PATH

# ─── Load ─────────────────────────────────────────────────────────────────────
def load_split(split_name, augmented=False):
    if augmented:
        path = os.path.join(DATASET_DIR, "train_keypoints_aug.csv")
    else:
        path = os.path.join(DATASET_DIR, f"{split_name}_keypoints.csv")
    df        = pd.read_csv(path)
    feat_cols = [c for c in df.columns if c not in ["label","filename"]]
    return df[feat_cols].values.astype(np.float32), df["label"].values

# ─── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(model, X, y_str, le, split_name):
    y_enc  = le.transform(y_str)
    y_pred_enc = model.predict(X)
    # XGBoost returns ints; sklearn models return strings
    if y_pred_enc.dtype in [np.int32, np.int64, np.int_]:
        y_pred_str = le.inverse_transform(y_pred_enc)
    else:
        y_pred_str = y_pred_enc
        y_pred_enc = le.transform(y_pred_str)

    top1 = accuracy_score(y_str, y_pred_str)
    top5 = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        top5  = top_k_accuracy_score(y_enc, proba, k=min(5, proba.shape[1]))
    print(f"  {split_name:6s} — Top-1: {top1*100:.2f}%"
          + (f"  Top-5: {top5*100:.2f}%" if top5 else ""))
    return top1, y_pred_str

# ─── Confusion matrix ─────────────────────────────────────────────────────────
def plot_cm(y_true, y_pred, model_name):
    from collections import Counter
    from sklearn.metrics import confusion_matrix
    top20   = [c for c, _ in Counter(y_true).most_common(20)]
    mask    = np.isin(y_true, top20)
    cm      = confusion_matrix(y_true[mask], y_pred[mask], labels=top20)
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(14,12))
    sns.heatmap(cm_norm, annot=False, xticklabels=top20, yticklabels=top20,
                cmap="Blues", ax=ax, vmin=0, vmax=1)
    ax.set_title(f"{model_name} — confusion matrix (top 20 classes)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8); plt.tight_layout()
    out = os.path.join(MODELS_DIR,
          f"confusion_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(out, dpi=150); plt.close()

# ─── Comparison chart ─────────────────────────────────────────────────────────
def plot_comparison(results):
    names = list(results.keys())
    x, w  = np.arange(len(names)), 0.25
    fig, ax = plt.subplots(figsize=(11,5))
    ax.bar(x-w, [results[n]["train"]*100 for n in names], w,
           label="Train", color="#1D9E75")
    ax.bar(x,   [results[n]["val"]  *100 for n in names], w,
           label="Val",   color="#378ADD")
    ax.bar(x+w, [results[n]["test"] *100 for n in names], w,
           label="Test",  color="#D85A30")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    ax.set_title("Model comparison — with augmented training data")
    ax.legend()
    for bars in ax.containers:
        ax.bar_label(bars, fmt="%.1f%%", fontsize=8, padding=2)
    plt.tight_layout()
    plt.savefig(os.path.join(MODELS_DIR,"model_comparison.png"), dpi=150)
    plt.close()
    print(f"Comparison chart saved.")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading data (augmented training set)...")
    X_train, y_train = load_split("train", augmented=True)
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")

    # Merge augmented train + val for final fit
    X_tv = np.vstack([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])
    print(f"  Train+Val: {X_tv.shape}  Test: {X_test.shape}")

    # Encode labels
    le = LabelEncoder()
    le.fit(y_tv)
    joblib.dump(le, os.path.join(MODELS_DIR,"label_encoder.pkl"))

    # Scale
    scaler = StandardScaler()
    X_tv_s = scaler.fit_transform(X_tv)
    X_te_s = scaler.transform(X_test)
    X_tr_s = scaler.transform(X_train)
    X_va_s = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)

    # Integer-encoded labels (for XGBoost + top-k scoring)
    y_tv_enc    = le.transform(y_tv)
    y_train_enc = le.transform(y_train)
    y_val_enc   = le.transform(y_val)
    y_test_enc  = le.transform(y_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            min_samples_split=5, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=20, min_samples_leaf=2,
            min_samples_split=5, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1),

        "SVM": SVC(
            kernel="rbf", C=50, gamma="scale",
            probability=True, class_weight="balanced", random_state=42),

        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1),
    }

    results    = {}
    best_acc   = 0
    best_name  = ""
    best_model = None

    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"  Training {name}...")
        print(f"{'='*55}")

        # Fit
        if name == "XGBoost":
            model.fit(X_tv_s, y_tv_enc)
        else:
            model.fit(X_tv_s, y_tv)

        # Evaluate on original (non-augmented) val & test
        train_acc, _      = evaluate(model, X_tr_s, y_train, le, "train")
        val_acc,   _      = evaluate(model, X_va_s, y_val,   le, "val")
        test_acc,  y_pred = evaluate(model, X_te_s, y_test,  le, "test")

        print(f"\n  Per-class report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        plot_cm(y_test, y_pred, name)

        results[name] = {"train": train_acc, "val": val_acc, "test": test_acc}

        if test_acc > best_acc:
            best_acc, best_name, best_model = test_acc, name, model

    # Save best
    joblib.dump(best_model, MODEL_PATH)
    meta = {"best_model": best_name,
            "test_accuracy": round(best_acc, 4),
            "num_classes": len(le.classes_)}
    with open(os.path.join(MODELS_DIR,"model_meta.json"),"w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*55}")
    print(f"  Best model : {best_name}  ({best_acc*100:.2f}%)")
    print(f"  Saved      → {MODEL_PATH}")
    print(f"{'='*55}")

    plot_comparison(results)
    print("\nPhase 3 complete!")

if __name__ == "__main__":
    main()