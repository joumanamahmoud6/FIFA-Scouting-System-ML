#  FIFA Player Scouting System
### A Unified Machine Learning Pipeline for Player Valuation & Performance Classification

![Python](https://img.shields.io/badge/Python-3.10-blue)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-1.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-latest-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  Overview
A two-phase ML project built on the [FIFA Players Dataset](https://drive.google.com/drive/folders/13Qb935zhUWRxn4yxMRqbsixXnEGLkhbC) (~19,667 players).
The system takes a player's raw stats and simultaneously predicts:
-  **Market Value** — regression on `Value Per M$`
-  **Performance Tier** — classification into `Low · Mid · High · Elite` (based on Overall_Rating percentiles)

Preprocessing is applied once and shared across all models: **Train/Test split (80/20) → One-Hot Encoding → StandardScaler → Outlier Handling.**
`Overall_Rating` is excluded from classification features to prevent data leakage.

---

##  Models & Methods

**Assignment 2 — Baselines**
- **Polynomial Regression** (degrees 1–4) with Ridge (L2) and Lasso (L1) regularization → best at degree 4, alpha 0.01, Test R² = 0.9728
- **Logistic Regression** with C sweep (L1 vs L2) → Mean CV Accuracy = 0.8043 ± 0.0027
- **Naïve Bayes** — GaussianNB / BernoulliNB / ComplementNB compared; GaussianNB unaffected by scaling by design
- **Cross-Validation** — 5-Fold for regression, Stratified 5-Fold for classification

**Assignment 3 — Advanced System**
- **KNN** (instance-based) · **SVM RBF kernel** (kernel-based) · **XGBoost** (tree-based)
- All tuned via **GridSearchCV (5-Fold)** — no defaults used
- Error diagnosis (train/test gap) confirms **Good Generalization** across all models
- **Stacking Ensemble** — KNN + SVM + XGBoost as base learners, Ridge meta-learner (regression) and Logistic Regression on `predict_proba` (classification), trained on **Out-Of-Fold predictions** to prevent leakage
- **Unified Pipeline** — single preprocessing step feeds both regression and classification stacks simultaneously
- **Stability Assessment** — CV fold scores confirm results are consistent, not split-dependent

---

## Final Results

| Task | Baseline (A2) | Stacking Ensemble (A3) |
|---|---|---|
| Regression — Test RMSE | 1.4950 | **1.3369 ✅** |
| Regression — Test R² | 0.9728 | **0.9665** |
| Classification — Accuracy | 0.8043 | **0.8701 ✅** |

---

##  Files
| File | Description |
|---|---|
| `FIFAanalysis_3.ipynb` | Complete notebook — all tasks from both assignments |
| `results.json` | Best hyperparameters and CV stability metrics for all models |
