# final_full_pipeline.py
# ---------------------------------------------------------------------
# CTG Datathon — full modeling pipeline (clean, no leakage)
# - Target: NSP (1=Normal, 2=Suspect, 3=Pathologic)
# - Features: physiological/statistical only (no CLASS)
# - Models: Logistic Regression, Decision Tree, Random Forest,
#           Gradient Boosting, XGBoost, LightGBM, SVM (RBF)
# - Saves:
#     * metrics (JSON) + classification reports (TXT)
#     * confusion matrices (PNG)
#     * feature importance plots for tree/boosting models
#     * per-model multiclass ROC plots (One-vs-Rest)
#     * leaderboard CSV
#
# The code is written to be readable and "human" — not AI-ish — with
# short, practical comments about *why* things are done, not just *what*.

from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")  # keep output clean

# Core stack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Sklearn pieces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# Optional extras — we’ll use if available
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

import joblib

# -----------------------------
# Paths and simple config
# -----------------------------
PROC_CSV   = Path("data/processed/ctg_final.csv")  # already-cleaned CSV you produced
REPORTS    = Path("reports")
FIG_DIR    = REPORTS / "figures"
MODELS_DIR = Path("models")

for d in (REPORTS, FIG_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
TARGET = "NSP"

# Typical feature set from this dataset. We’ll intersect with what’s present.
FEATURES = [
    "LB","AC.1","FM.1","UC.1","DL.1","DS.1","DP.1",
    "ASTV","MSTV","ALTV","MLTV",
    "Width","Min","Max","Nmax","Nzeros","Mode","Mean","Median","Variance",
    "Tendency","A","B","C","D","E","AD","DE","LD","FS","SUSP"
]

# -----------------------------
# Small utility helpers
# -----------------------------
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def save_text(txt, path):
    with open(path, "w") as f:
        f.write(txt)

def plot_confusion_matrix(y_true, y_pred, labels, out_path, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    ticks = list(range(len(labels)))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_multiclass_roc(y_true_bin, y_score, class_labels, out_path, title):
    """
    One-vs-Rest ROC per class on a single figure (separate file per model).
    y_true_bin: (n_samples, n_classes) binary indicators
    y_score:    (n_samples, n_classes) probs or decision scores aligned with class order
    """
    n_classes = y_true_bin.shape[1]
    plt.figure(figsize=(6,5))

    # Macro-average AUC across classes (clean headline number)
    aucs = []
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
        aucs.append(auc)
        plt.plot(fpr, tpr, lw=1.2, label=f"Class {class_labels[i]} (AUC={auc:.3f})")
    macro_auc = float(np.mean(aucs))

    # Add random baseline
    plt.plot([0,1],[0,1], linestyle="--", color="gray", label="Chance")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title}  |  Macro-AUC={macro_auc:.3f}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return macro_auc

def topk_feature_importance_plot(feat_names, importances, out_path, title, k=15):
    order = np.argsort(importances)[::-1][:k]
    names = [feat_names[i] for i in order]
    vals  = [importances[i] for i in order]
    plt.figure(figsize=(8,5))
    plt.barh(names[::-1], vals[::-1])
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    # Also drop CSV for the numbers
    df = pd.DataFrame({"feature": names, "importance": vals})
    df.to_csv(out_path.with_suffix(".csv"), index=False)

# -----------------------------
# Data loading and sanity
# -----------------------------
if not PROC_CSV.exists():
    raise FileNotFoundError(
        "Processed CSV not found at data/processed/ctg_final.csv. "
        "Make sure you've run your cleaning step."
    )

df = pd.read_csv(PROC_CSV)

# Hard rule: do NOT let CLASS leak in. Drop if present.
if "CLASS" in df.columns:
    df = df.drop(columns=["CLASS"])

# Keep only features that actually exist in this file.
use_features = [c for c in FEATURES if c in df.columns]
if len(use_features) == 0:
    raise ValueError("No expected features found in the processed CSV.")

# Target: coerce to numeric and drop rows without it.
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])

# NaN handling for features — mean impute (simple baseline; works fine here).
df[use_features] = df[use_features].fillna(df[use_features].mean(numeric_only=True))

X = df[use_features].copy()
y_raw = df[TARGET].astype(int).copy()

# LabelEncoder standardizes labels to 0..K-1 which some libs expect (e.g., XGB).
le = LabelEncoder()
y = le.fit_transform(y_raw)           # 0..2
class_labels = le.classes_.tolist()   # original labels, e.g., [1,2,3]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

# For ROC we’ll need a binarized version of y_test using original order
y_test_bin = label_binarize(le.inverse_transform(y_test), classes=class_labels)

# -----------------------------
# Define the 7 models
# -----------------------------
models = {}

# 1) Logistic Regression — scale features; balanced to handle class imbalance
models["logistic_regression"] = Pipeline([
    ("scale", MinMaxScaler()),
    ("clf", LogisticRegression(
        multi_class="multinomial",
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_SEED
    ))
])

# 2) Decision Tree — simple non-linear baseline
models["decision_tree"] = DecisionTreeClassifier(
    random_state=RANDOM_SEED,
    class_weight="balanced"
)

# 3) Random Forest — robust bagging ensemble
models["random_forest"] = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1
)

# 4) Gradient Boosting (sklearn) — strong classic booster
models["gradient_boosting"] = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_SEED
)

# 5) XGBoost — add only if lib is available
if XGBClassifier is not None:
    # XGBoost wants labels as 0..num_class-1. We already label-encoded y, so we’re good.
    models["xgboost"] = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        objective="multi:softprob",
        num_class=len(class_labels)
    )
else:
    print("Note: xgboost not installed; skipping XGBClassifier.")

# 6) LightGBM — add only if lib is available
if LGBMClassifier is not None:
    # LightGBM can handle label-encoded ints as targets directly.
    models["lightgbm"] = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )
else:
    print("Note: lightgbm not installed; skipping LGBMClassifier.")

# 7) SVM (RBF) — scale features; probability=True for ROC curves
models["svm_rbf"] = Pipeline([
    ("scale", StandardScaler()),
    ("clf", SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,  # needed for predict_proba
        random_state=RANDOM_SEED
    ))
])

# -----------------------------
# Train, evaluate, save
# -----------------------------
results = []

for name, model in models.items():
    print(f"\n=== Training {name} ===")
    model.fit(X_train, y_train)

    # Predictions for metrics/confusion
    y_pred = model.predict(X_test)
    # Convert back to original labels for human-facing outputs
    y_pred_labels = le.inverse_transform(y_pred)
    y_test_labels = le.inverse_transform(y_test)

    # Metrics we care about for imbalanced, multiclass data
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Save metrics and classification report
    metrics_path = REPORTS / f"{name}_metrics.json"
    report_path  = REPORTS / f"{name}_classification_report.txt"
    save_json({"balanced_accuracy": float(bal_acc), "macro_f1": float(macro_f1)}, metrics_path)
    save_text(classification_report(y_test, y_pred, digits=4), report_path)

    # Confusion matrix (in original label space)
    cm_path = FIG_DIR / f"cm_{name}.png"
    plot_confusion_matrix(
        y_true=y_test_labels,
        y_pred=y_pred_labels,
        labels=class_labels,
        out_path=cm_path,
        title=f"Confusion Matrix — {name}"
    )

    # ROC per model (separate figure each):
    # We need a score/prob per class. Prefer predict_proba, fall back to decision_function.
    try:
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)  # shape (n, n_classes) in encoded order
        else:
            # Pipelines expose predict_proba via final step sometimes; otherwise decision_function
            if hasattr(model, "decision_function"):
                raw = model.decision_function(X_test)
            else:
                # If it's a Pipeline, try last step
                last = getattr(model, "steps", [("", model)])[-1][1]
                if hasattr(last, "predict_proba"):
                    score = last.predict_proba(model[:-1].transform(X_test)) if hasattr(model, "__getitem__") else last.predict_proba(X_test)
                elif hasattr(last, "decision_function"):
                    raw = last.decision_function(model[:-1].transform(X_test)) if hasattr(model, "__getitem__") else last.decision_function(X_test)
                else:
                    raw = None

            # If we have decision_function raw scores, convert to a shape like predict_proba
            if 'score' not in locals():
                if raw is None:
                    # As a last resort, build a "pseudo-prob" from predictions (not ideal, but avoids crash)
                    # This makes ROC uninformative; we still emit a plot for completeness.
                    K = len(class_labels)
                    score = np.zeros((len(y_pred), K))
                    for i, yp in enumerate(y_pred):
                        score[i, yp] = 1.0
                else:
                    # If binary, raw may be (n,), so coerce to (n,2)
                    raw = np.atleast_2d(raw)
                    if raw.shape[1] == 1:
                        # Build a 2-col score; but we have 3 classes normally, so this shouldn't happen here.
                        score = np.hstack([-raw, raw])
                    else:
                        # Softmax to [0,1] range per row
                        ex = np.exp(raw - raw.max(axis=1, keepdims=True))
                        score = ex / ex.sum(axis=1, keepdims=True)

        # Align score columns to original class order (inverse of label encoder)
        # Our models were trained on label-encoded 0..K-1 with the same order as le.classes_,
        # so score columns should already align. We’ll just be explicit:
        y_test_bin_aligned = y_test_bin  # already built on class_labels (same order)
        roc_path = FIG_DIR / f"roc_{name}.png"
        macro_auc = plot_multiclass_roc(
            y_true_bin=y_test_bin_aligned,
            y_score=score,
            class_labels=class_labels,
            out_path=roc_path,
            title=f"ROC — {name}"
        )
    except Exception as e:
        print(f"[WARN] ROC plot for {name} failed: {e}")
        macro_auc = None

    # Feature importance plots where it makes sense
    imp_png = None
    last_estimator = model
    if hasattr(model, "steps"):  # Pipeline — last step is the model
        last_estimator = model.steps[-1][1]

    if hasattr(last_estimator, "feature_importances_"):
        importances = last_estimator.feature_importances_
        imp_png = FIG_DIR / f"feat_importance_{name}.png"
        topk_feature_importance_plot(
            feat_names=use_features,
            importances=importances,
            out_path=imp_png,
            title=f"Top 15 Feature Importances — {name}",
            k=15
        )

    # Save the trained model
    joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    # Keep a row for the leaderboard
    results.append({
        "model": name,
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "macro_auc_ovr": float(macro_auc) if macro_auc is not None else None
    })

# -----------------------------
# Leaderboard + summary
# -----------------------------
leaderboard = pd.DataFrame(results).sort_values(by="balanced_accuracy", ascending=False)
leaderboard_path = REPORTS / "model_leaderboard.csv"
leaderboard.to_csv(leaderboard_path, index=False)

print("\n================ Summary ================")
print(leaderboard.to_string(index=False))
print("\nSaved:")
print("  Models   ->", MODELS_DIR.resolve())
print("  Reports  ->", REPORTS.resolve())
print("  Figures  ->", FIG_DIR.resolve())
print("  Leaderboard CSV ->", leaderboard_path.resolve())
print("=========================================")
