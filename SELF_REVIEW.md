# SELF_REVIEW — Iterate, Diagnose, Defend (Week 06 · Day 4)

Requirement-by-requirement check of every today-task item against the actual
delivered artifacts. Verification source: `test_pipeline.py`
(72 checks passing) plus notebook output inspection.

## Today's tasks

| # | Task | Delivered where | Verified |
|---|---|---|---|
| 1 | Train unconstrained decision tree with RMSE and R² | Notebook Section 5 | ✓ (depth=20, train RMSE=0.00, test RMSE=9.84) |
| 2 | Compare training vs test RMSE (overfitting demo) | Notebook Section 5 | ✓ (gap=9.84 points — clear overfitting) |
| 3 | Constrained tree (max_depth=3) compared to unconstrained and linear | Notebook Section 5 | ✓ (RMSE=7.10, competitive with linear) |
| 4 | Random forest regression with RMSE and R² | Notebook Section 6 | ✓ (RMSE=7.40, R²=0.48) |
| 5 | Feature importances compared to linear regression coefficients | Notebook Section 6 | ✓ (study_hours dominant in both) |
| 6 | Random forest classification with accuracy/precision/recall/F1 | Notebook Section 6 | ✓ (acc=0.73, F1=0.80) |
| 7 | Regression comparison table (all 5 models) | Notebook Section 7 | ✓ (sorted by RMSE) |
| 8 | Classification comparison table (all 3 models) | Notebook Section 7 | ✓ (all 4 metrics) |
| 9 | Written defense of winner with specific numbers | Notebook Section 7 markdown | ✓ (linear regression wins regression, logistic wins classification) |
| 10 | Error analysis: 5 worst regression predictions with pattern | Notebook Section 8 | ✓ (unusual feature combinations) |
| 11 | Error analysis: misclassified rows with pattern | Notebook Section 8 | ✓ (near-boundary students) |
| 12 | Calibration curve for classifier | Notebook Section 9 | ✓ (5 bins, reasonable calibration) |
| 13 | Calibration interpretation sentence | Notebook Section 9 markdown | ✓ |
| 14 | Reproducible notebook (all seeds fixed) | All sections | ✓ (seed=21, random_state=42) |
| 15 | Self-audit table verifying all numbers | Notebook Section 11 | ✓ (all metrics match) |
| 16 | Honest limitations documented | Notebook Section 12 | ✓ |
| 17 | Evaluation report for Friday's defense | evaluation_report.md | ✓ |

## Capstone required content (in order)

| Requirement | Present? | Notes |
|---|---|---|
| Unconstrained decision tree trained | ✓ | depth=20, 421 leaves |
| Training vs test RMSE compared (overfitting) | ✓ | Train RMSE=0.00, Test RMSE=9.84 |
| Constrained tree (max_depth=3) trained | ✓ | RMSE=7.10, R²=0.52 |
| Constrained vs unconstrained vs linear compared | ✓ | Table in Section 7 |
| Random forest regression (n_estimators=200) | ✓ | RMSE=7.40, R²=0.48 |
| Feature importances extracted and compared | ✓ | study_hours dominant in both forest and linear |
| Random forest classification (n_estimators=200) | ✓ | acc=0.73, F1=0.80 |
| Regression comparison table (all models) | ✓ | 5 models, sorted by RMSE |
| Classification comparison table (all models) | ✓ | 3 models, all 4 metrics |
| Winner defended with specific numbers | ✓ | Linear RMSE=7.06 < Forest RMSE=7.40 |
| Error analysis: regression residuals | ✓ | Top 5 worst, pattern identified |
| Error analysis: classification misclassified | ✓ | Pattern near decision boundary |
| Calibration curve | ✓ | 5 bins, interpretation provided |
| Self-audit table | ✓ | All metrics verified |
| Technical summary | ✓ | evaluation_report.md |
| Honest limitations | ✓ | 6 limitations documented |
| Survives Restart Kernel & Run All | ✓ | Zero error cells |
| 72-check test suite passes | ✓ | All 72 checks passing |

## Pre-submission checklist applied

1. Every number in the write-up comes from a live variable / f-string — ✓
2. Self-audit table matches all claimed numbers — ✓
3. Every decision has "why" reasoning — ✓
4. Every chart has title + labeled axes — ✓
5. Charts are matched to questions — ✓
6. Honest limitations documented — ✓
7. Evaluation report understandable by non-technical reader — ✓
8. Git: feature branch, meaningful commits, pushed — ✓

## Adversarial Self-Questions

- **What looks correct but might be wrong?**
  The unconstrained tree's perfect training RMSE (0.00) might look like a good
  result, but it's actually overfitting — the test RMSE of 9.84 proves it.
  This is the intended lesson, not a bug.

- **What would break if input changed?**
  The test suite hard-codes expected ranges (e.g., distinction rate 50-80%).
  On data with very different distributions, some checks would need updating.

- **What could a skeptic question?**
  The random forest underperforms linear regression on this dataset. A skeptic
  might think we chose bad hyperparameters. Mitigated by noting that default
  settings were used and the dataset has known linear relationships.

- **What did we NOT do?**
  No cross-validation (single split used). No hyperparameter tuning for random
  forest. All stated as limitations.

## Honest Self-Assessment

The notebook is complete and passes all 72 verification checks. The key lesson
— that a fancier algorithm is not automatically a better model — is clearly
demonstrated: linear regression (RMSE=7.06) beats random forest (RMSE=7.40) on
this dataset. The error analysis identifies concrete patterns in worst predictions.
The calibration curve shows reasonable calibration with some deviation at extremes.
The evaluation report provides standalone prose suitable for Friday's defense.
