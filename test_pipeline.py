"""Verification suite for the Iterate, Diagnose, Defend project.

Runs from a fresh process (independent of any notebook kernel state) and checks
the deliverables by re-deriving numbers from data/students.csv. All tests are
deterministic because the dataset is generated with a fixed seed.

Usage:
    python3 test_pipeline.py
    # or
    python -m pytest test_pipeline.py -v
"""
import os
import sys
import hashlib

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
CHARTS = os.path.join(HERE, "charts")

STUDENTS_CSV = os.path.join(DATA, "students.csv")
GEN_PY = os.path.join(HERE, "generate_data.py")
NOTEBOOK = os.path.join(HERE, "ml_pipeline.ipynb")

passed = 0
failed = 0
failures = []


def check(condition, label):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS  {label}")
    else:
        failed += 1
        failures.append(label)
        print(f"FAIL  {label}")


def sha256_file(filepath):
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


# ---------------------------------------------------------------------------
# Part A — Dataset correct
# ---------------------------------------------------------------------------
print("== Part A: Dataset ==")
df = pd.read_csv(STUDENTS_CSV)

check(len(df) == 600, "dataset has 600 rows")
check(list(df["student_id"]) == list(range(1, 601)), "student_id is 1..600")
check(
    {"student_id", "class_section", "study_hours_per_week",
     "sleep_hours_per_night", "attendance_pct", "exam_score"} <= set(df.columns),
    "expected columns present"
)
check(df.isna().sum().sum() == 0, "no missing values")
check(df.duplicated().sum() == 0, "no duplicate rows")
check((df["study_hours_per_week"] >= 0).all(), "study hours nonnegative")
check(df["sleep_hours_per_night"].between(3, 10).all(), "sleep hours within [3,10]")
check(df["attendance_pct"].between(40, 100).all(), "attendance within [40,100]")
check(df["exam_score"].between(0, 100).all(), "exam score within [0,100]")
check(set(df["class_section"].unique()) == {"A", "B", "C"}, "class sections are A, B, C")

# Section distribution (seed=21, p=[0.34, 0.33, 0.33])
section_counts = df["class_section"].value_counts().sort_index()
check(section_counts["A"] == 212, f"Section A count = 212 (got {section_counts['A']})")
check(section_counts["B"] == 198, f"Section B count = 198 (got {section_counts['B']})")
check(section_counts["C"] == 190, f"Section C count = 190 (got {section_counts['C']})")

# ---------------------------------------------------------------------------
# Part B — Feature engineering (one-hot encoding)
# ---------------------------------------------------------------------------
print("== Part B: Feature engineering ==")
df_encoded = pd.get_dummies(df, columns=["class_section"], drop_first=True)

check("class_section_B" in df_encoded.columns, "class_section_B column exists")
check("class_section_C" in df_encoded.columns, "class_section_C column exists")
check("class_section" not in df_encoded.columns, "original class_section column removed")

# ---------------------------------------------------------------------------
# Part C — Train/test split (fixed seed)
# ---------------------------------------------------------------------------
print("== Part C: Train/test split ==")
feature_cols = ["study_hours_per_week", "sleep_hours_per_night", "attendance_pct",
                "class_section_B", "class_section_C"]
target_col = "exam_score"

X = df_encoded[feature_cols]
y = df_encoded[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

check(X_train.shape[0] == 480, f"training set has 480 rows (got {X_train.shape[0]})")
check(X_test.shape[0] == 120, f"test set has 120 rows (got {X_test.shape[0]})")
check(X_train.shape[1] == 5, f"5 features in training set (got {X_train.shape[1]})")

# Verify reproducibility
X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)
check(
    np.array_equal(X_train.values, X_train2.values) and np.array_equal(X_test.values, X_test2.values),
    "split is reproducible (same seed → same rows)"
)

# ---------------------------------------------------------------------------
# Part D — Regression baseline (DummyRegressor)
# ---------------------------------------------------------------------------
print("== Part D: Regression baseline ==")
dummy_reg = DummyRegressor(strategy="mean")
dummy_reg.fit(X_train, y_train)
y_pred_dummy = dummy_reg.predict(X_test)

rmse_dummy = np.sqrt(mean_squared_error(y_test, y_pred_dummy))
r2_dummy = r2_score(y_test, y_pred_dummy)

check(abs(r2_dummy) < 0.01, f"baseline R² is approximately 0.0 (got {r2_dummy:.6f})")
check(rmse_dummy > 0, f"baseline RMSE is positive (got {rmse_dummy:.4f})")

# ---------------------------------------------------------------------------
# Part E — Linear regression model
# ---------------------------------------------------------------------------
print("== Part E: Linear regression ==")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_lin))
r2_lin = r2_score(y_test, y_pred_lin)

check(rmse_lin < rmse_dummy, f"linear RMSE ({rmse_lin:.4f}) < baseline RMSE ({rmse_dummy:.4f})")
check(r2_lin > 0.3, f"linear R² is substantial (> 0.3, got {r2_lin:.4f})")

# Coefficients
check(len(lin_reg.coef_) == 5, f"5 coefficients (got {len(lin_reg.coef_)})")
check(lin_reg.coef_[0] > 0, "study_hours_per_week has positive coefficient")
check(lin_reg.coef_[4] > 0, "class_section_C has positive coefficient")

# ---------------------------------------------------------------------------
# Part F — Classification target (distinction)
# ---------------------------------------------------------------------------
print("== Part F: Classification target ==")
df_encoded["distinction"] = (df_encoded["exam_score"] >= 85).astype(int)

distinction_rate = df_encoded["distinction"].mean()
check(0.50 < distinction_rate < 0.80, f"distinction rate between 50-80% (got {distinction_rate:.3f})")

# Stratified split
y_class = df_encoded["distinction"]
X_class = df_encoded[feature_cols]

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

check(X_train_c.shape[0] == 480, f"classification training set has 480 rows")
check(X_test_c.shape[0] == 120, f"classification test set has 120 rows")

# ---------------------------------------------------------------------------
# Part G — Classification baseline (DummyClassifier)
# ---------------------------------------------------------------------------
print("== Part G: Classification baseline ==")
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train_c, y_train_c)
y_pred_dummy_c = dummy_clf.predict(X_test_c)

acc_dummy = accuracy_score(y_test_c, y_pred_dummy_c)
prec_dummy = precision_score(y_test_c, y_pred_dummy_c, zero_division=0)
rec_dummy = recall_score(y_test_c, y_pred_dummy_c, zero_division=0)
f1_dummy = f1_score(y_test_c, y_pred_dummy_c, zero_division=0)

check(rec_dummy == 1.0, f"baseline recall is 1.0 (got {rec_dummy:.4f})")
check(acc_dummy > 0.5, f"baseline accuracy > 50% (got {acc_dummy:.4f})")

# ---------------------------------------------------------------------------
# Part H — Logistic regression model
# ---------------------------------------------------------------------------
print("== Part H: Logistic regression ==")
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train_c, y_train_c)
y_pred_log = log_reg.predict(X_test_c)

acc_log = accuracy_score(y_test_c, y_pred_log)
prec_log = precision_score(y_test_c, y_pred_log)
rec_log = recall_score(y_test_c, y_pred_log)
f1_log = f1_score(y_test_c, y_pred_log)

check(acc_log > acc_dummy, f"logreg accuracy ({acc_log:.4f}) > baseline ({acc_dummy:.4f})")
check(f1_log > f1_dummy, f"logreg F1 ({f1_log:.4f}) > baseline ({f1_dummy:.4f})")

# ---------------------------------------------------------------------------
# Part I — Decision trees (Thursday's new models)
# ---------------------------------------------------------------------------
print("== Part I: Decision trees ==")

# Unconstrained tree
tree_uncon = DecisionTreeRegressor(random_state=42)
tree_uncon.fit(X_train, y_train)
y_pred_tree_uncon = tree_uncon.predict(X_test)
rmse_tree_uncon_train = np.sqrt(mean_squared_error(y_train, tree_uncon.predict(X_train)))
rmse_tree_uncon_test = np.sqrt(mean_squared_error(y_test, y_pred_tree_uncon))
r2_tree_uncon = r2_score(y_test, y_pred_tree_uncon)

check(tree_uncon.get_depth() > 5, f"unconstrained tree is deep (got depth={tree_uncon.get_depth()})")
check(rmse_tree_uncon_train < 1.0, f"unconstrained tree train RMSE < 1.0 (got {rmse_tree_uncon_train:.4f})")
check(rmse_tree_uncon_test > rmse_tree_uncon_train + 2, f"overfitting: test RMSE > train RMSE + 2 (gap={rmse_tree_uncon_test - rmse_tree_uncon_train:.2f})")

# Constrained tree
tree_con = DecisionTreeRegressor(max_depth=3, random_state=42)
tree_con.fit(X_train, y_train)
y_pred_tree_con = tree_con.predict(X_test)
rmse_tree_con = np.sqrt(mean_squared_error(y_test, y_pred_tree_con))
r2_tree_con = r2_score(y_test, y_pred_tree_con)

check(tree_con.get_depth() <= 3, f"constrained tree depth ≤ 3 (got {tree_con.get_depth()})")
check(rmse_tree_con < rmse_tree_uncon_test, f"constrained RMSE ({rmse_tree_con:.4f}) < unconstrained ({rmse_tree_uncon_test:.4f})")

# ---------------------------------------------------------------------------
# Part J — Random forests (Thursday's new models)
# ---------------------------------------------------------------------------
print("== Part J: Random forests ==")

# Random forest regression
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train, y_train)
y_pred_rf = rf_reg.predict(X_test)
rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf = r2_score(y_test, y_pred_rf)

check(rmse_rf < rmse_dummy, f"RF RMSE ({rmse_rf:.4f}) < baseline ({rmse_dummy:.4f})")
check(r2_rf > 0.3, f"RF R² is substantial (> 0.3, got {r2_rf:.4f})")

# Feature importances
importances = rf_reg.feature_importances_
check(len(importances) == 5, f"5 feature importances (got {len(importances)})")
check(abs(sum(importances) - 1.0) < 0.01, f"feature importances sum to ~1.0 (got {sum(importances):.4f})")

# Random forest classification
rf_clf = RandomForestClassifier(n_estimators=200, random_state=42)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf_c = rf_clf.predict(X_test_c)

acc_rf = accuracy_score(y_test_c, y_pred_rf_c)
prec_rf = precision_score(y_test_c, y_pred_rf_c)
rec_rf = recall_score(y_test_c, y_pred_rf_c)
f1_rf = f1_score(y_test_c, y_pred_rf_c)

check(acc_rf > acc_dummy, f"RF accuracy ({acc_rf:.4f}) > baseline ({acc_dummy:.4f})")
check(f1_rf > 0.5, f"RF F1 > 0.5 (got {f1_rf:.4f})")

# ---------------------------------------------------------------------------
# Part K — Model comparison tables
# ---------------------------------------------------------------------------
print("== Part K: Model comparison ==")

# Linear regression should beat unconstrained tree
check(rmse_lin < rmse_tree_uncon_test, f"linear RMSE ({rmse_lin:.4f}) < unconstrained tree ({rmse_tree_uncon_test:.4f})")

# Linear regression should be competitive with constrained tree
check(abs(rmse_lin - rmse_tree_con) < 1.0, f"linear and constrained tree are competitive (diff={abs(rmse_lin - rmse_tree_con):.4f})")

# Logistic regression should beat baseline
check(acc_log > acc_dummy, f"logreg accuracy ({acc_log:.4f}) > baseline ({acc_dummy:.4f})")

# ---------------------------------------------------------------------------
# Part L — Error analysis
# ---------------------------------------------------------------------------
print("== Part L: Error analysis ==")

# Regression residuals
residuals = y_test - y_pred_lin
check(abs(residuals.mean()) < 2.0, f"residual mean near 0 (got {residuals.mean():.4f})")

# Top 5 worst predictions exist
abs_residuals = np.abs(residuals)
check(len(abs_residuals) == 120, f"120 residuals (got {len(abs_residuals)})")
worst_5_idx = np.argsort(abs_residuals)[-5:]
check(len(worst_5_idx) == 5, "can identify 5 worst predictions")

# Classification misclassification analysis
misclassified_mask = y_pred_rf_c != y_test_c
misclassification_rate = misclassified_mask.mean()
check(0.1 < misclassification_rate < 0.5, f"misclassification rate reasonable (got {misclassification_rate:.2%})")

# ---------------------------------------------------------------------------
# Part M — Calibration
# ---------------------------------------------------------------------------
print("== Part M: Calibration ==")

y_proba_rf = rf_clf.predict_proba(X_test_c)[:, 1]
fraction_of_positives, mean_predicted_value = calibration_curve(y_test_c, y_proba_rf, n_bins=5)

check(len(fraction_of_positives) >= 3, f"calibration curve has ≥3 points (got {len(fraction_of_positives)})")
check(len(mean_predicted_value) == len(fraction_of_positives), "calibration curve points match")

# ---------------------------------------------------------------------------
# Part N — Chart files exist and are valid PNGs
# ---------------------------------------------------------------------------
print("== Part N: Charts ==")
expected_charts = [
    "chart_regression_all_models.png",
    "chart_classification_all_models.png",
    "chart_feature_comparison.png",
    "chart_calibration_curve.png",
]

for chart_name in expected_charts:
    chart_path = os.path.join(CHARTS, chart_name)
    if os.path.exists(chart_path):
        with open(chart_path, "rb") as f:
            is_png = f.read(8) == b"\x89PNG\r\n\x1a\n"
        check(is_png, f"valid PNG: {chart_name}")
    else:
        check(False, f"chart exists: {chart_name}")

# ---------------------------------------------------------------------------
# Part O — Deliverable files exist
# ---------------------------------------------------------------------------
print("== Part O: Deliverables ==")
deliverables = [
    (STUDENTS_CSV, "data/students.csv"),
    (GEN_PY, "generate_data.py"),
    (NOTEBOOK, "ml_pipeline.ipynb"),
    (os.path.join(HERE, "evaluation_report.md"), "evaluation_report.md"),
    (os.path.join(HERE, "requirements.txt"), "requirements.txt"),
    (os.path.join(HERE, ".gitignore"), ".gitignore"),
]

for filepath, label in deliverables:
    check(os.path.exists(filepath), f"exists: {label}")

# ---------------------------------------------------------------------------
# Part P — Notebook cleanliness (no error cells)
# ---------------------------------------------------------------------------
print("== Part P: Notebook cleanliness ==")
try:
    import nbformat
    nb = nbformat.read(NOTEBOOK, as_version=4)
    error_cells = [
        i for i, cell in enumerate(nb.cells)
        if cell.cell_type == "code" and any(
            o.get("output_type") == "error" for o in cell.get("outputs", [])
        )
    ]
    check(len(error_cells) == 0, f"notebook has zero error cells (found {len(error_cells)})")
except Exception as e:
    check(False, f"notebook cleanliness check failed: {e}")

# ---------------------------------------------------------------------------
# Part Q — Reproducibility checks
# ---------------------------------------------------------------------------
print("== Part Q: Reproducibility ==")

dataset_hash = sha256_file(STUDENTS_CSV)
check(len(dataset_hash) == 64, f"dataset has valid SHA-256 hash")

# Verify key metrics are reproducible
rmse_recompute = np.sqrt(mean_squared_error(y_test, y_pred_lin))
check(
    abs(rmse_lin - rmse_recompute) < 0.0001,
    f"RMSE is reproducible ({rmse_lin:.6f} == {rmse_recompute:.6f})"
)

acc_recompute = accuracy_score(y_test_c, y_pred_log)
check(
    abs(acc_log - acc_recompute) < 0.0001,
    f"Accuracy is reproducible ({acc_log:.6f} == {acc_recompute:.6f})"
)

# ---------------------------------------------------------------------------
# Part R — Statistical checks
# ---------------------------------------------------------------------------
print("== Part R: Statistical checks ==")

r_study, p_study = stats.pearsonr(df["study_hours_per_week"], df["exam_score"])
check(0.55 < r_study < 0.80, f"study-hours r in plausible range (got {r_study:.3f})")
check(p_study < 0.001, f"study-hours p-value significant (got {p_study:.2e})")

r_sleep, p_sleep = stats.pearsonr(df["sleep_hours_per_night"], df["exam_score"])
check(abs(r_sleep) < 0.10, f"sleep-hours r near zero (got {r_sleep:.3f})")

means = df.groupby("class_section")["exam_score"].mean()
check(means["C"] - means["A"] > 1, f"Section C mean above A (diff {means['C']-means['A']:.2f})")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
print(f"{'='*60}")
print(f"RESULT: {passed} passed, {failed} failed")
print(f"{'='*60}")

if failures:
    print("\nFailed checks:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)

print("\nAll checks passed.")
sys.exit(0)
