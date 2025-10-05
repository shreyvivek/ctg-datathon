# Datathon 2025 – Lifeline: CTG Analysis for Fetal Health Classification

## 👥 Team Information
- **Team ID:** TM-202
- **Team Members:**
  - Vivek Shrey (Team Leader)
  - Ravisankar Navisshna
  - Beh Seng Kiat

---

## Project Overview

This project was developed for **Datathon 2025: Lifeline**, focusing on **predictive fetal health classification** using **Cardiotocography (CTG) data**.  
We built a complete end-to-end pipeline to clean, process, analyze, and model CTG signals to predict fetal condition categories (Normal / Suspect / Pathological).

Our objective is to support early detection of potential fetal distress and improve clinical decision-making through interpretable machine learning models.

---

## Workflow Summary

1. **Data Cleaning & Preprocessing** – We extracted and cleaned 2000+ CTG records, handled missing data, and prepared meaningful features.
2. **Exploratory Data Analysis (EDA)** – Performed statistical analysis and visualizations to understand feature importance and class distributions.
3. **Model Development** – Built and evaluated multiple ML models including:
   - Logistic Regression
   - Decision Tree
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - Support Vector Machine (SVM)
4. **Evaluation & Insights** – Compared models using accuracy, precision, recall, F1-score, and ROC-AUC to identify the most robust approach. Established that **Random Forest** gave us the best results (More Details in INSIGHTS.md)
5. **Interpretation & Impact** – Highlighted key contributing factors for fetal classification and provided insights for potential clinical deployment.

---

## How to Run

1. **Set Up the Environment** - Create and activate a virtual environment and install dependencies:
  - `python3 -m venv .venv`
  - `source .venv/bin/activate`
  - `pip install -r requirements.txt`
2. **Generate the 23 feature training file** - This ensures the dataset matches the competition’s required input schema.
  - `python scripts/build_23_feature_view.py`
3. **Train the model** - This trains a model using only the 23 features specified by the requirements of the Datathon. It will output the trained weights (`models/best_model.joblib`) and key metrics.
  - `python scripts/train.py --train_csv data/processed/ctg_23train.csv --label_col NSP`

  - Expected output: Balanced accuracy ~0.82, Macro F1 ~0.85
4. **Run inference on a new sample** - This uses the trained model to predict NSP on new input data.
  - `python scripts/inference.py --weights models/best_model.joblib --input_csv sample_input.csv --out predictions.csv`

  - Output: A `predictions.csv file` containing the predicted NSP values.
5. **Reproduce full results** - If you want to replicate the full pipeline performance (including feature engineering, model selection, and tuning), run:
  - `python scripts/final_full_pipeline.py` : to train and evaluate the 7 models. 
  - `python scripts/step6_tune_top2_fast.py` : to fine-tune top-performing models (Random Forest & XGBoost).
  - `python scripts/step6_finalize.py` : to generate final metrics, plots, and insights.
  
  - Expected results:
    - Baseline balanced accuracy ~0.88
    - Tuned accuracy ~0.89
    - Metrics and plots will be saved in `reports/`
6. **Check Outputs** - 
  - Model Leaderboard : `reports/model_leaderboard.csv`
  - Final metrics : `reports/FINAL_metrics.json`
  - Plots : `reports/figures/`
  - Insights : `reports/INSIGHTS.md`





