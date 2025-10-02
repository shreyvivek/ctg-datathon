from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
import joblib

# --- Paths ---
PROC = Path("data/processed/ctg_final.csv")
FIG  = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
REP  = Path("reports"); REP.mkdir(parents=True, exist_ok=True)
MOD  = Path("models")

# --- Config ---
TARGET = "NSP"
RANDOM_SEED = 42
FEATURES = [
    "LB","AC.1","FM.1","UC.1","DL.1","DS.1","DP.1",
    "ASTV","MSTV","ALTV","MLTV","Width","Min","Max",
    "Nmax","Nzeros","Mode","Mean","Median","Variance",
    "Tendency","A","B","C","D","E","AD","DE","LD","FS","SUSP"
]

FINALISTS = [
    ("random_forest", MOD / "random_forest.joblib"),
    ("xgboost",       MOD / "xgboost.joblib"),
]

# --- Utils ---
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def plot_cm(y_true, y_pred, labels, out_png, title, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues", interpolation="nearest")
    plt.title(title)
    plt.xlabel("Predicted"); plt.ylabel("True")
    ticks = list(range(len(labels)))
    plt.xticks(ticks, labels); plt.yticks(ticks, labels)
    for (i,j), v in np.ndenumerate(cm):
        txt = f"{v:.2f}" if normalize else str(v)
        plt.text(j, i, txt, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()

def plot_multiclass_roc(y_true_bin, y_score, class_labels, out_png, title):
    plt.figure(figsize=(6,5))
    aucs = []
    for i, lab in enumerate(class_labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_score[:, i])
        aucs.append(auc)
        plt.plot(fpr, tpr, lw=1.2, label=f"{lab} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1], "--", color="gray", label="Chance")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(title + f" | Macro-AUC={np.mean(aucs):.3f}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300); plt.close()
    return float(np.mean(aucs))

# --- Load data ---
df = pd.read_csv(PROC)
if "CLASS" in df.columns:  # no leakage
    df = df.drop(columns=["CLASS"])

use_features = [c for c in FEATURES if c in df.columns]
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df[use_features] = df[use_features].fillna(df[use_features].mean(numeric_only=True))

X = df[use_features].copy()
y_raw = df[TARGET].astype(int).copy()

le = LabelEncoder()
y = le.fit_transform(y_raw)             # 0..K-1
class_labels = le.classes_.tolist()     # e.g. [1,2,3]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)
y_test_lbl = le.inverse_transform(y_test)                  # original labels for plots
y_test_bin = label_binarize(y_test_lbl, classes=class_labels)

# --- Evaluate finalists ---
rows = []
for name, path in FINALISTS:
    if not path.exists():
        print(f"[Skip] {name} model not found at {path}")
        continue

    model = joblib.load(path)

    # predictions
    y_pred = model.predict(X_test)
    y_pred_lbl = le.inverse_transform(y_pred)

    # metrics
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report   = classification_report(y_test, y_pred, digits=4, output_dict=True)
    pd.DataFrame(report).T.to_csv(REP / f"{name}_per_class_metrics.csv")

    # confusion matrices
    plot_cm(y_test_lbl, y_pred_lbl, class_labels, FIG / f"cm_{name}_raw_step6.png", f"Confusion Matrix — {name} (raw)", normalize=False)
    plot_cm(y_test_lbl, y_pred_lbl, class_labels, FIG / f"cm_{name}_norm_step6.png", f"Confusion Matrix — {name} (normalized)", normalize=True)

    # ROC scores (predict_proba preferred)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(X_test)
    else:
        # Pipeline? grab last step if needed
        last = getattr(model, "steps", [("", model)])[-1][1]
        if hasattr(last, "predict_proba"):
            score = last.predict_proba(model[:-1].transform(X_test)) if hasattr(model, "__getitem__") else last.predict_proba(X_test)
        else:
            # fallback: decision_function -> softmax
            raw = last.decision_function(X_test) if hasattr(last, "decision_function") else model.decision_function(X_test)
            ex = np.exp(raw - raw.max(axis=1, keepdims=True))
            score = ex / ex.sum(axis=1, keepdims=True)

    macro_auc = plot_multiclass_roc(
        y_true_bin=y_test_bin,
        y_score=score,
        class_labels=class_labels,
        out_png=FIG / f"roc_{name}_step6.png",
        title=f"ROC — {name}"
    )

    save_json({"balanced_accuracy": float(bal_acc),
               "macro_f1": float(macro_f1),
               "macro_auc_ovr": float(macro_auc)},
              REP / f"{name}_step6_metrics.json")

    rows.append({"model": name, "balanced_accuracy": bal_acc, "macro_f1": macro_f1, "macro_auc_ovr": macro_auc})

# summary
if rows:
    lb = pd.DataFrame(rows).sort_values("balanced_accuracy", ascending=False)
    lb.to_csv(REP / "step6_top2_leaderboard.csv", index=False)
    print("\nStep6 Top-2 Leaderboard:")
    print(lb.to_string(index=False))
else:
    print("No finalist models found. Make sure you've trained and saved them first.")
