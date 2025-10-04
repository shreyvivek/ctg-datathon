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