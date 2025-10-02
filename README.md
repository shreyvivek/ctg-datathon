# CTG Datathon — Final Submission

## 📍 Project Overview
This solution predicts **fetal state (`NSP`)** from Cardiotocographic (CTG) signal-derived features.  
The model classifies each case into one of three categories:

- **1 – Normal:** healthy fetal condition  
- **2 – Suspect:** potentially abnormal, requires further observation  
- **3 – Pathologic:** high-risk, immediate clinical attention needed

The pipeline is designed to be:
- **Reproducible:** deterministic splits (seed = 42), stratified CV, and version-controlled scripts.
- **Leak-free:** `CLASS` is excluded from training features.
- **Clinically relevant:** performance metrics emphasize **balanced accuracy** and **macro-F1** to account for class imbalance.

---

## 📂 Folder Structure
submission/
│
├── README.md <- This file
├── FINAL_metrics.json <- Key metrics for the winning model
├── INSIGHTS.md <- Full technical narrative + explanations
├── EXEC_SUMMARY.md <- 1-page executive summary (for slides/report)
│
├── reports/
│ ├── step6_tuned_fast_leaderboard.csv
│ ├── *_FINAL_per_class_metrics.csv
│ ├── *FINAL_test_predictions.csv
│ └── figures/
│ ├── cm<winner>FINAL_raw.png
│ ├── cm<winner>FINAL_norm.png
│ ├── roc<winner>FINAL.png
│ └── feat_importance<winner>_FINAL.png
│
├── models/
│ ├── random_forest_tuned_fast.joblib
│ └── xgboost_tuned_fast.joblib
│
├── scripts/
│ ├── final_full_pipeline.py
│ ├── step6_tune_top2_fast.py
│ └── step6_finalize.py
│
└── data/
└── processed/
└── ctg_final.csv

---

## 🧪 Quick Reproduction (Mac / Linux)

### 1️⃣ Create and activate virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
# or minimal install:
pip install numpy pandas scikit-learn matplotlib joblib xgboost lightgbm
python scripts/final_full_pipeline.py

This will:

Clean and preprocess data

Train 7 baseline models

Save performance reports and metrics

Optional: Hyperparameter Tuning (~3 min)

python scripts/step6_tune_top2_fast.py

Tunes the top-2 ensemble models (Random Forest and XGBoost) with randomized search and 3-fold CV.

Final Evaluation & Plots

python scripts/step6_finalize.py

Generates:

Final confusion matrices

ROC curves

Feature importance plots

FINAL_metrics.json and INSIGHTS.md

Key Metrics (from FINAL_metrics.json)
Metric	Description
Balanced Accuracy	Handles class imbalance fairly
Macro F1	Weighted average of per-class F1
Macro AUC (OvR)	One-vs-Rest ROC AUC across all classes

Notes & Assumptions

Leakage prevention: CLASS is excluded (it’s derived after classification).

Metrics focus: Balanced accuracy and macro-F1 handle imbalanced labels better than raw accuracy.

Seed fixed (42): Ensures reproducibility.

Clinical trade-off: Ensembles prioritize recall for minority classes (Suspect, Pathologic) even if precision dips slightly.

Next Steps (Future Work)

Calibrate decision thresholds for better Pathologic recall.

Experiment with sequence models (e.g. 1D-CNN, LSTM) on raw time-series.

Use SHAP or LIME for interpretability and clinician-facing explanations.

