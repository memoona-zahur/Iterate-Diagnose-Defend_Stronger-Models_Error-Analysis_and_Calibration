"""Dataset generator for the Iterate, Diagnose, Defend project.

Generates the exact students dataset from the assignment spec (seed=21, n=600).
This file is the single source of truth for the dataset — do not modify.

Usage:
    python3 generate_data.py
    # Creates data/students.csv (600 × 6)
"""
import numpy as np
import pandas as pd

OUTPUT = "data/students.csv"

rng = np.random.default_rng(seed=21)
n = 600

class_section = rng.choice(["A", "B", "C"], size=n, p=[0.34, 0.33, 0.33])
study_hours = rng.normal(10, 3.5, size=n).clip(0, None).round(1)
sleep_hours = rng.normal(7, 1.2, size=n).clip(3, 10).round(1)
attendance_pct = rng.normal(85, 10, size=n).clip(40, 100).round(1)

noise = rng.normal(0, 8, size=n)
section_bonus = pd.Series(class_section).map({"A": 0, "B": 0, "C": 4}).values
exam_score = (50 + 2.6 * study_hours + 0.15 * attendance_pct + section_bonus + noise).clip(0, 100).round(1)

students = pd.DataFrame({
    "student_id": np.arange(1, n + 1),
    "class_section": class_section,
    "study_hours_per_week": study_hours,
    "sleep_hours_per_night": sleep_hours,
    "attendance_pct": attendance_pct,
    "exam_score": exam_score,
})

students.to_csv(OUTPUT, index=False)
print(f"Generated {OUTPUT}: {students.shape[0]} rows × {students.shape[1]} columns")
print(f"Columns: {list(students.columns)}")
print(f"Section distribution: {dict(students['class_section'].value_counts().sort_index())}")
print(f"Exam score range: [{students['exam_score'].min()}, {students['exam_score'].max()}]")
