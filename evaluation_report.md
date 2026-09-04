# Evaluation Report — Iterate, Diagnose, Defend

*Week 06 · Thursday · A complete ML pipeline with model comparison, error analysis, and calibration.*

## What the Task Was

We built a machine learning pipeline to predict student exam performance from study habits, sleep, attendance, and class section. The pipeline includes two tasks:

1. **Regression**: Predict the actual exam score (a continuous number)
2. **Classification**: Predict whether a student will achieve distinction (score ≥ 85)

The dataset contains 600 synthetic students with known relationships built in, generated with a fixed random seed (seed=21) for full reproducibility.

## The Baseline

Before building any real model, we established "dumb baselines" — the simplest possible predictors:

- **Regression baseline**: Always predict the mean exam score (~88 points). RMSE = 10.30, R² ≈ 0
- **Classification baseline**: Always predict "distinction" (the majority class). Accuracy = 65%, but this is misleading — the model learns nothing and can't distinguish students

Every real model must beat these baselines to be worth using.

## Models Compared

We trained and evaluated five regression models and three classification models:

### Regression Models

| Model | RMSE | R² |
|-------|------|-----|
| Baseline (DummyRegressor) | 10.30 | -0.0096 |
| Linear Regression | **7.06** | **0.5257** |
| Decision Tree (unconstrained) | 9.84 | 0.0782 |
| Decision Tree (max_depth=3) | 7.10 | 0.5197 |
| Random Forest (200 trees) | 7.40 | 0.4785 |

### Classification Models

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Baseline (DummyClassifier) | 0.6500 | 0.6500 | 1.0000 | 0.7879 |
| Logistic Regression | **0.7667** | **0.8125** | **0.8333** | **0.8228** |
| Random Forest (200 trees) | 0.7250 | 0.7711 | 0.8205 | 0.7950 |

## Our Chosen Final Models

### Regression: Linear Regression

**We chose linear regression as our best model for predicting exam scores.** It achieves RMSE=7.06 and R²=0.53, which outperforms every other model we tested — including the random forest (RMSE=7.40) and both decision trees.

This result is the central lesson of today's work: **a fancier algorithm is not automatically a better model.** The unconstrained decision tree overfit badly (training RMSE ≈ 0 but test RMSE = 9.84). Even the constrained tree and random forest couldn't beat linear regression on this dataset. The data has genuinely linear relationships (study hours coefficient = 2.03 points/hour, Section C bonus = 3.13 points), and a linear model captures them more efficiently than tree-based methods.

### Classification: Logistic Regression

**We chose logistic regression as our best model for predicting distinction.** It achieves accuracy=0.77, precision=0.81, recall=0.83, and F1=0.82 — all superior to the random forest (accuracy=0.73, F1=0.80). The logistic regression model is also far more interpretable: we can directly read which features drive predictions from its coefficients.

## One Error Analysis Finding

The five rows the linear regression model gets most wrong share a common pattern: **they have unusual combinations of features** — students with high study hours but low attendance, or vice versa. The average study hours for the worst-predicted 5 students is 8.8 (vs. 10.3 for all test students), and their average attendance is 78.7% (vs. 85.4%). This suggests the model struggles most when features point in conflicting directions — a natural limitation of a linear model that assumes additive effects.

For classification, the misclassified rows tend to be students near the distinction boundary (exam score close to 85), where the signal is weakest. The model's hardest cases are the ambiguous ones, which is expected behavior.

## One Calibration Finding

The random forest classifier's calibration curve shows that its predicted probabilities roughly correspond to actual success rates. The model is reasonably well-calibrated overall, though it shows some deviation at extreme probabilities — at low confidence (predicted ≈ 0.08), the actual positive rate is 0%, and at moderate confidence (predicted ≈ 0.30), the actual positive rate is 50%. This means the model's stated confidence can be taken at face value for high-probability predictions but should be treated with caution for borderline cases.

## What We Could NOT Do

- **Causation**: We cannot say studying more *causes* higher scores — only that they're correlated
- **Generalization**: The model was trained on 600 synthetic students; real students might differ
- **Feature limitations**: Only 5 features; many real factors are omitted
- **Single split**: We used one fixed train/test split instead of cross-validation
- **Hyperparameter tuning**: Random forest used default settings; optimization might improve performance

## Reproducibility

Every random operation in this pipeline is seeded (seed=21 for data generation, random_state=42 for train/test split and all models). The notebook survives Restart Kernel and Run All with zero error cells. All metrics have been independently verified in a self-audit table.

---

*This report is the direct rehearsal for Friday's written and oral defense.*
