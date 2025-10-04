from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import balanced_accuracy_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

PROC = Path("data/processed/ctg_final.csv")
MODELS = Path("models"); MODELS.mkdir(parents=True, exist_ok=True)
REPORTS = Path("reports"); REPORTS.mkdir(parents=True, exist_ok=True)

TARGET = "NSP"
RAND = 42
FEATURES = [
    "LB","AC.1","FM.1","UC.1","DL.1","DS.1","DP.1",
    "ASTV","MSTV","ALTV","MLTV","Width","Min","Max",
    "Nmax","Nzeros","Mode","Mean","Median","Variance",
    "Tendency","A","B","C","D","E","AD","DE","LD","FS","SUSP"
]

df = pd.read_csv(PROC)
if "CLASS" in df.columns:
    df = df.drop(columns=["CLASS"])

use_features = [c for c in FEATURES if c in df.columns]
df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
df = df.dropna(subset=[TARGET])
df[use_features] = df[use_features].fillna(df[use_features].mean(numeric_only=True))

X = df[use_features].copy()
y_raw = df[TARGET].astype(int).copy()

le = LabelEncoder()
y = le.fit_transform(y_raw)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RAND, stratify=y
)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RAND)
scoring = "balanced_accuracy"

results_rows = []

rf = RandomForestClassifier(random_state=RAND, class_weight="balanced", n_jobs=-1)
rf_space = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
}
rf_search = RandomizedSearchCV(
    rf, rf_space, n_iter=18, cv=cv, scoring=scoring, n_jobs=-1, random_state=RAND, verbose=0
)
rf_search.fit(X_train, y_train)
rf_best = rf_search.best_estimator_
y_pred = rf_best.predict(X_test)
rf_metrics = {
    "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
    "best_params": rf_search.best_params_
}
with open(REPORTS / "rf_tuned_fast_metrics.json", "w") as f: json.dump(rf_metrics, f, indent=2)
with open(REPORTS / "rf_tuned_fast_classification_report.txt", "w") as f:
    f.write(classification_report(y_test, y_pred, digits=4))
joblib.dump(rf_best, MODELS / "random_forest_tuned_fast.joblib")
results_rows.append({"model":"random_forest_tuned_fast",
                     "balanced_accuracy": rf_metrics["balanced_accuracy"],
                     "macro_f1": rf_metrics["macro_f1"]})

if XGBClassifier is not None:
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        n_jobs=-1,
        random_state=RAND,
        tree_method="hist",         
        eval_metric="mlogloss"
    )
    xgb_space = {
        "n_estimators": [250, 400, 600],
        "max_depth": [4, 5, 6],
        "learning_rate": [0.03, 0.05, 0.1],
        "subsample": [0.7, 0.85, 1.0],
        "colsample_bytree": [0.7, 0.85, 1.0],
        "min_child_weight": [1, 2, 4]
    }
    xgb_search = RandomizedSearchCV(
        xgb, xgb_space, n_iter=20, cv=cv, scoring=scoring, n_jobs=-1, random_state=RAND, verbose=0
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    y_pred = xgb_best.predict(X_test)
    xgb_metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "best_params": xgb_search.best_params_
    }
    with open(REPORTS / "xgb_tuned_fast_metrics.json", "w") as f: json.dump(xgb_metrics, f, indent=2)
    with open(REPORTS / "xgb_tuned_fast_classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, digits=4))
    joblib.dump(xgb_best, MODELS / "xgboost_tuned_fast.joblib")
    results_rows.append({"model":"xgboost_tuned_fast",
                         "balanced_accuracy": xgb_metrics["balanced_accuracy"],
                         "macro_f1": xgb_metrics["macro_f1"]})
else:
    print("Skipping XGBoost fast tuning (xgboost not available).")

if results_rows:
    lb = pd.DataFrame(results_rows).sort_values("balanced_accuracy", ascending=False)
    lb.to_csv(REPORTS / "step6_tuned_fast_leaderboard.csv", index=False)
    print("\nStep6 Tuned (fast) Leaderboard:")
    print(lb.to_string(index=False))
