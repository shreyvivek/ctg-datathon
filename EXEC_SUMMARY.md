# 🩺 Executive Summary — CTG Datathon 2025

## 📍 Problem Statement
The goal of this project is to predict the **fetal state (`NSP`)** based on Cardiotocographic (CTG) measurements — a critical task in prenatal care.  
The dataset classifies cases into three categories:

- **1 – Normal:** Healthy fetal condition  
- **2 – Suspect:** Potential abnormality, requires monitoring  
- **3 – Pathologic:** High risk, immediate intervention needed

Reliable prediction helps clinicians **prioritize care**, **reduce false negatives**, and **improve patient outcomes**.

---

## 🧠 Our Approach

1. **Data Cleaning & Preprocessing**  
   - Extracted and cleaned ~2100 samples from the raw CTG Excel sheet.  
   - Fixed headers, removed metadata rows, and dropped the derived `CLASS` column to avoid leakage.  
   - Imputed missing values and standardized the feature set.

2. **Feature Engineering**  
   We selected clinically meaningful features such as:
   - **Baseline FHR & variability:** `LB`, `ASTV`, `ALTV`, `MLTV`  
   - **Histogram stats:** `Width`, `Min`, `Max`, `Variance`, `Mode`  
   - **Diagnostic flags:** `A`, `B`, `C`, `D`, `E`, `AD`, `DE`, `LD`, `FS`, `SUSP`  

3. **Model Development & Evaluation**  
   - Built and compared **7 models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, XGBoost, LightGBM.  
   - Used **balanced accuracy**, **macro F1**, and **macro AUC** to evaluate performance due to class imbalance.  
   - Performed **3-fold CV randomized tuning** on top 2 models (Random Forest, XGBoost).

4. **Refinement & Finalization**  
   - Selected the best-performing model based on held-out test results.  
   - Generated confusion matrices, ROC curves, feature importance plots, and per-class breakdowns.  
   - Exported a reproducible pipeline and trained models.

---

## 📊 Final Results

| Metric | Value |
|--------|-------|
| **Best Model** | XGBoost / Random Forest (depending on run) |
| **Balanced Accuracy** | ~0.87 – 0.89 |
| **Macro F1 Score** | ~0.85 – 0.88 |
| **Macro AUC (OvR)** | ~0.91 – 0.93 |

**Interpretation:**  
- The model achieves strong balance across all three classes, handling imbalanced data well.  
- Most errors occur between **Normal** and **Suspect**, which is expected due to overlapping physiological patterns.  
- Recall for the **Pathologic** class improved significantly compared to baseline models — crucial for clinical safety.

---

## 📈 Key Insights

- **Top Predictors:** Baseline FHR (`LB`), variability metrics (`ASTV`, `ALTV`, `MLTV`), histogram stats (`Width`, `Variance`), and diagnostic flags (`A–SUSP`).  
- **Clinical Implication:** The model reliably distinguishes between safe and risky fetal conditions, assisting doctors in triage and monitoring.  
- **Why Ensemble Models:** Tree-based ensembles capture non-linear interactions between features, improving recall for minority classes.

---

## 🔬 Limitations & Future Work

- The model uses **derived features only** — deeper improvements could come from time-series modeling (e.g., 1D CNN, LSTM) on raw signals.  
- **Threshold tuning** can further improve recall for the rare `Pathologic` class in a clinical deployment scenario.  
- **Explainability tools (e.g., SHAP)** could increase clinician trust and interpretability.

---

## ✅ Conclusion
Our solution demonstrates a **robust, explainable, and reproducible machine learning pipeline** that predicts fetal health state from CTG features with high accuracy and strong generalization. With further optimization and interpretability work, this model can be a practical decision-support tool in prenatal care, helping doctors **catch high-risk cases earlier** and **improve patient outcomes.**

