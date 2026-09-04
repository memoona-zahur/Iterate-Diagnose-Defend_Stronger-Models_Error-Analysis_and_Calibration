# Technical Summary — Iterate, Diagnose, Defend (Week 06 · Day 4)

*Plain-language write-up for a non-technical reader.*

## What we built

We extended Wednesday's machine learning pipeline with more advanced models, error analysis, and calibration checks. The pipeline predicts student exam performance from study habits, sleep, attendance, and class section.

## What we found

### 1. A fancier algorithm is not automatically better

We trained five regression models and four classification models. The results surprised us:

**For predicting exam scores:**
- Linear regression (RMSE=7.06) beat the random forest (RMSE=7.40)
- The unconstrained decision tree overfit badly — it memorized the training data (RMSE=0.00) but failed on new data (RMSE=9.84)
- Constraining the tree (max_depth=3) fixed the overfitting but didn't beat linear regression

**For predicting distinction students:**
- Logistic regression (F1=0.82) beat the random forest (F1=0.80)
- Both beat the baseline, but the simpler model won

### 2. Where the model errs

The five rows the linear regression model gets most wrong share a pattern: they have unusual combinations of features — students with high study hours but low attendance, or vice versa. The model struggles when features point in conflicting directions.

### 3. Is the model's confidence trustworthy?

The logistic regression classifier's calibration curve shows reasonable calibration — its predicted probabilities roughly correspond to actual success rates. It is well-calibrated at the extremes but mildly under-confident in the mid-range probabilities, so borderline predictions should be treated with slight caution.

## Why this matters

The key lesson is that **model complexity doesn't guarantee better performance**. On this dataset with its known linear relationships, a simple linear regression outperforms sophisticated ensemble methods. This is why we compare models on evidence instead of reputation.

## What checks we ran

- **72 automated checks** validating every metric, dataset property, and deliverable
- **Self-audit table** independently recomputing all claimed numbers
- **Reproducible pipeline** with fixed random seeds (seed=21, random_state=42)
- **Restart Kernel and Run All** verified with zero errors

## Honest limitations

- **Correlation ≠ causation**: We cannot say studying more *causes* higher scores
- **Synthetic data**: Real students might have different patterns
- **Single split**: Cross-validation would give more robust estimates
- **No hyperparameter tuning**: Random forest used default settings
- **Limited features**: Only 5 features; many real factors are omitted
