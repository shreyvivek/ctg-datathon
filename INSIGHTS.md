# CTG Datathon — Final Model Summary

**Winner:** `random_forest_tuned_fast`

## Headline Metrics

- Balanced Accuracy: **0.8972**
- Macro F1: **0.8934**
- Macro AUC (OvR): **0.9825**

## Why this model
- Best overall generalization on imbalanced classes (balanced accuracy).
- Stable across folds during tuning (small variance expected for tree ensembles).

## Per-class behavior (from normalized confusion matrix)
- Row = true class; values are recall per class.
- Common confusions: Suspect ↔ Normal (clinical borderline). Pathologic is rarer; recall improves with ensemble models.

## Top signals driving predictions (feature importance)
- See figure and CSV for the top 10–15 features.
- Typically strong: **LB (baseline FHR)**, **ASTV/ALTV/MLTV (short/long-term variability)**,
  **histogram stats (Width/Min/Max/Mode/Mean/Median/Variance)**, and diagnostic flags (**A–SUSP**).

## Modeling choices that matter
- **No leakage**: excluded `CLASS` entirely.
- **Stratified split, fixed seed (42)** for reproducibility.
- **Balanced metrics** (balanced accuracy + macro-F1) due to class imbalance.

## Limitations & next steps
- Temporal dynamics are summarized into features; a sequence model could definitely capture more signal.
- Threshold tuning for clinic-specific trade-offs (recall for Pathologic vs. precision) could be explored.
