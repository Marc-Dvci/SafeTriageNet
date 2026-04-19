"""
Build a self-contained Kaggle notebook for SafeTriageNet.

The notebook produced here has no external src/ dependency and runs on
Kaggle, where the triagegeist data is expected at /kaggle/input/triagegeist.

Run from the SafeTriageNet directory:
    python build/build_kaggle_notebook.py
"""

import json
import os
import re
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SRC = REPO / "src"
OUT_NB = REPO / "notebooks" / "safetriagenet.ipynb"


def load_src(filename: str) -> str:
    """Load a source file and strip its module docstring + top-level imports.

    Uses ast to find exact line ranges of top-level imports so multi-line
    `from x import (...)` statements are fully removed.
    """
    import ast

    content = (SRC / filename).read_text(encoding="utf-8")
    tree = ast.parse(content)
    lines = content.splitlines()

    # Collect 1-based line ranges to drop (imports + module docstring)
    drop_ranges: List[Tuple[int, int]] = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            drop_ranges.append((node.lineno, node.end_lineno))
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
            and node is tree.body[0]
        ):
            drop_ranges.append((node.lineno, node.end_lineno))

    drop = set()
    for start, end in drop_ranges:
        for ln in range(start, end + 1):
            drop.add(ln)

    cleaned: List[str] = []
    for i, line in enumerate(lines, start=1):
        if i in drop:
            continue
        if line.strip().startswith("warnings.filterwarnings"):
            continue
        cleaned.append(line)

    while cleaned and not cleaned[0].strip():
        cleaned.pop(0)
    return "\n".join(cleaned).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Notebook content
# ---------------------------------------------------------------------------

CELLS = []


def md(text: str):
    CELLS.append(new_markdown_cell(text.strip() + "\n"))


def code(text: str):
    CELLS.append(new_code_cell(text.strip() + "\n"))


# --- Title --------------------------------------------------------------

md("""
# SafeTriageNet: When Getting It Wrong Matters More Than Getting It Right

*Safety-Aware Multimodal Triage with Informative Missingness and Asymmetric Clinical Cost*

**Triagegeist Competition — Laitinen-Fredriksson Foundation**

---

Emergency triage is a high-stakes ranking problem disguised as a classification problem. A
triage nurse is not only assigning a label; they are deciding who can wait and who cannot.
In that setting, all mistakes are not equal. Over-triaging a stable patient is inefficient,
but under-triaging a critically ill patient can delay life-saving care.

**SafeTriageNet** treats triage acuity prediction as a patient-safety problem, not a pure
accuracy contest. The goal is not the single most aggressive classifier on paper — it is a
model that keeps accuracy competitive while shifting the error profile away from the most
dangerous misses.

### Three pillars

1. **Informative Missingness** — missing vitals are treated as a clinical signal, not just
   as a nuisance to impute away.
2. **Multimodal feature set** — structured intake, light-weight complaint-text heuristics,
   and comorbidity history are engineered into a single modeling table.
3. **Safety-aware selection** — an asymmetric clinical cost matrix, sample weights for
   rare high-acuity classes, and entropy-triggered conservative shifting toward safer
   predictions.

This notebook is self-contained: it runs end-to-end on Kaggle without any external src/
dependency and produces `submission.csv` in the working directory.
""")

# --- Section 1: Setup --------------------------------------------------

md("## 1. Setup & Imports")

code("""
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report,
)

import lightgbm as lgb
import xgboost as xgb

SEED = 42
np.random.seed(SEED)

plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.figsize": (14, 8),
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "figure.dpi": 100,
})
PALETTE = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"]
ESI_NAMES = {1: "Resuscitation", 2: "Emergent", 3: "Urgent",
             4: "Less Urgent", 5: "Non-urgent"}

print("SafeTriageNet -- setup complete.")
""")

# --- Section 2: Data Loading -------------------------------------------

md("""
## 2. Data Loading

The notebook tries the Kaggle competition data path first (`/kaggle/input/triagegeist`),
then standard Kaggle dataset mirrors, then a few local fallbacks so it can also be run
outside Kaggle. The required files are:

- `train.csv` — 80,000 training visits with the `triage_acuity` label
- `test.csv`  — 20,000 test visits
- `chief_complaints.csv` — free-text chief complaint per patient
- `patient_history.csv` — structured comorbidity flags per patient
- `sample_submission.csv` — required output format
""")

code('''
REQUIRED_FILES = [
    "train.csv",
    "test.csv",
    "chief_complaints.csv",
    "patient_history.csv",
    "sample_submission.csv",
]


def _cwd_safe() -> Path:
    try:
        return Path.cwd()
    except Exception:
        return Path(".")


def resolve_data_dir() -> Path:
    """Find the triagegeist data folder across Kaggle and local layouts."""
    candidates: List[Path] = []

    env_dir = os.environ.get("TRIAGEGEIST_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    # Kaggle competition / dataset mounts (standard Kaggle layout)
    candidates.extend([
        Path("/kaggle/input/triagegeist"),
        Path("/kaggle/input/triagegeist-data"),
        Path("/kaggle/input/triagegeist-synthetic"),
    ])

    # If a Kaggle dataset was mounted under a different name, sweep /kaggle/input
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        for sub in sorted(kaggle_root.iterdir()):
            if sub.is_dir() and sub not in candidates:
                candidates.append(sub)

    # Local fallbacks
    here = _cwd_safe()
    for extra in [
        here / "triagegeist",
        here.parent / "triagegeist",
        here.parent.parent / "triagegeist",
        Path("./triagegeist"),
        Path("../triagegeist"),
        Path("../../triagegeist"),
    ]:
        candidates.append(extra)

    seen: set = set()
    for cand in candidates:
        try:
            cand = cand.resolve()
        except Exception:
            continue
        if cand in seen:
            continue
        seen.add(cand)
        if cand.is_dir() and all((cand / name).exists() for name in REQUIRED_FILES):
            return cand

    raise FileNotFoundError(
        "Could not find the triagegeist data. Expected the five CSVs under "
        "/kaggle/input/triagegeist (Kaggle) or a local ./triagegeist/ folder. "
        "Set TRIAGEGEIST_DATA_DIR to override."
    )


DATA_DIR = resolve_data_dir()
print(f"Using data directory: {DATA_DIR}")

train_raw = pd.read_csv(DATA_DIR / "train.csv")
test_raw = pd.read_csv(DATA_DIR / "test.csv")
complaints = pd.read_csv(DATA_DIR / "chief_complaints.csv")
history = pd.read_csv(DATA_DIR / "patient_history.csv")
sample_sub = pd.read_csv(DATA_DIR / "sample_submission.csv")

print(f"\\n  Train set:         {train_raw.shape[0]:>7,} records x {train_raw.shape[1]} columns")
print(f"  Test set:          {test_raw.shape[0]:>7,} records x {test_raw.shape[1]} columns")
print(f"  Chief complaints:  {complaints.shape[0]:>7,} records")
print(f"  Patient history:   {history.shape[0]:>7,} records")
print(f"  Sample submission: {sample_sub.shape[0]:>7,} records")

print("\\n  Target distribution (triage_acuity):")
print(train_raw["triage_acuity"].value_counts().sort_index().to_string())
''')

# --- Section 3: Clinical safety module --------------------------------

md("""
## 3. Clinical Safety Module

This section defines the **asymmetric clinical cost matrix** and the clinical safety
metrics reported alongside standard ML metrics. Under-triage of ESI-1/2 patients is
penalized much more heavily than over-triage of ESI-4/5 patients.
""")

# Safety module (inline)
safety_body = load_src("safety.py")
# Drop the final CUSTOM LOSS block header that has a sklearn import note
code(safety_body)

# --- Section 4: Feature engineering module ----------------------------

md("""
## 4. Feature Engineering Module

Clinical-grade feature engineering: informative missingness, derived physiologic indices,
lightweight complaint-text heuristics, comorbidity burdens, and temporal context.
""")

features_body = load_src("features.py")
code(features_body)

# --- Section 5: Modeling module ---------------------------------------

md("""
## 5. Modeling Module

5-fold stratified LightGBM and XGBoost base learners, stacking meta-learner with its own
out-of-fold evaluation, and the prediction pipeline that feeds test data through the
full stack.
""")

models_body = load_src("models.py")
code(models_body)

# --- Section 6: EDA ----------------------------------------------------

md("""
## 6. Exploratory Data Analysis

Two questions drive the EDA:

1. **Is the class imbalance big enough to matter for evaluation?** Yes — ESI-1 is the
   rarest class, and a single aggregate accuracy score hides what happens on the patients
   who need triage support the most.
2. **Is missingness informative?** The dataset description claims so. We test it with a
   chi-square association test per vital.

All figures in this section are saved alongside the notebook.
""")

code('''
OUTPUT_DIR = Path(os.environ.get("SAFETRIAGE_OUTPUT_DIR", "."))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -- 6a: Target distribution + cost matrix --
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
counts = train_raw["triage_acuity"].value_counts().sort_index()
colors = [PALETTE[i] for i in range(5)]
bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=1.5)
axes[0].set_xlabel("ESI Acuity Level", fontweight="bold")
axes[0].set_ylabel("Patient Count", fontweight="bold")
axes[0].set_title("Triage Acuity Distribution (Training Set)")
for bar, count in zip(bars, counts.values):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2.0,
        bar.get_height() + 200,
        f"{count:,}\\n({count / len(train_raw) * 100:.1f}%)",
        ha="center", va="bottom", fontsize=10, fontweight="bold",
    )
for esi, name in ESI_NAMES.items():
    axes[0].text(esi, -max(counts.values) * 0.06, name,
                 ha="center", fontsize=8, style="italic", color="gray")

im = axes[1].imshow(CLINICAL_COST_MATRIX, cmap="YlOrRd", aspect="auto")
axes[1].set_xticks(range(5))
axes[1].set_yticks(range(5))
axes[1].set_xticklabels([f"ESI-{i + 1}" for i in range(5)])
axes[1].set_yticklabels([f"ESI-{i + 1}" for i in range(5)])
axes[1].set_xlabel("Predicted Acuity", fontweight="bold")
axes[1].set_ylabel("Actual Acuity", fontweight="bold")
axes[1].set_title("Asymmetric Clinical Cost Matrix")
for i in range(5):
    for j in range(5):
        color = "white" if CLINICAL_COST_MATRIX[i, j] > 8 else "black"
        axes[1].text(j, i, f"{CLINICAL_COST_MATRIX[i, j]:.1f}",
                     ha="center", va="center", fontsize=12, fontweight="bold", color=color)
plt.colorbar(im, ax=axes[1], label="Clinical Cost")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig1_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code('''
# -- 6b: Informative missingness --
print("\\n-- Informative missingness analysis --")
vital_cols_present = [c for c in CORE_VITALS if c in train_raw.columns]
rows = []
for col in vital_cols_present:
    for acuity in sorted(train_raw["triage_acuity"].unique()):
        subset = train_raw[train_raw["triage_acuity"] == acuity]
        rows.append({"vital": col, "acuity": acuity,
                     "missing_rate": subset[col].isna().mean()})
if "pain_score" in train_raw.columns:
    for acuity in sorted(train_raw["triage_acuity"].unique()):
        subset = train_raw[train_raw["triage_acuity"] == acuity]
        rows.append({"vital": "pain_score", "acuity": acuity,
                     "missing_rate": (subset["pain_score"] == -1).mean()})

miss_by_acuity = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
pivot = miss_by_acuity.pivot(index="vital", columns="acuity", values="missing_rate")
sns.heatmap(
    pivot, annot=True, fmt=".3f", cmap="RdYlGn_r", ax=axes[0],
    linewidths=1, linecolor="white",
    xticklabels=[f"ESI-{i}" for i in pivot.columns],
    cbar_kws={"label": "Missing Rate"},
)
axes[0].set_title("Vital Sign Missing Rate by ESI Acuity Level\\n(Higher = More Missing)",
                  fontweight="bold")
axes[0].set_xlabel("Triage Acuity (ESI)", fontweight="bold")
axes[0].set_ylabel("")

n_vitals_check = train_raw[vital_cols_present].notna().sum(axis=1)
doc_tmp = n_vitals_check / len(vital_cols_present)
doc_comp_by_acuity = doc_tmp.groupby(train_raw["triage_acuity"]).mean()
bars = axes[1].bar(doc_comp_by_acuity.index, doc_comp_by_acuity.values,
                   color=colors, edgecolor="white", linewidth=1.5)
axes[1].set_xlabel("ESI Acuity Level", fontweight="bold")
axes[1].set_ylabel("Mean Documentation Completeness", fontweight="bold")
axes[1].set_title("Vital Sign Documentation Completeness by Acuity\\n(Higher = More Vitals Recorded)",
                  fontweight="bold")
axes[1].set_ylim(0, 1.05)
for bar, val in zip(bars, doc_comp_by_acuity.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2.0,
                 bar.get_height() + 0.01, f"{val:.3f}",
                 ha="center", va="bottom", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig2_missingness_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nChi-square test: missingness x acuity")
significant, non_significant = [], []
for col in vital_cols_present:
    missing_flag = train_raw[col].isna().astype(int)
    contingency = pd.crosstab(missing_flag, train_raw["triage_acuity"])
    chi2, p_val, _, _ = stats.chi2_contingency(contingency)
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    print(f"  {col:<22} chi2={chi2:>10.1f}  p={p_val:.2e}  {sig}")
    (significant if p_val < 0.05 else non_significant).append(col)
print(f"\\n  -> {len(significant)}/{len(vital_cols_present)} core vitals show significant acuity-linked missingness.")
if non_significant:
    print(f"     Non-significant: {', '.join(non_significant)}")
''')

code('''
# -- 6c: Vital distributions by acuity --
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
vital_plot_cols = ["heart_rate", "systolic_bp", "respiratory_rate",
                   "spo2", "temperature_c", "gcs_total"]
for idx, col in enumerate(vital_plot_cols):
    ax = axes[idx // 3][idx % 3]
    if col not in train_raw.columns:
        continue
    for acuity in [1, 2, 3, 4, 5]:
        data = train_raw[train_raw["triage_acuity"] == acuity][col].dropna()
        if len(data) > 0:
            ax.hist(data, bins=40, alpha=0.4, label=f"ESI-{acuity}",
                    color=PALETTE[acuity - 1], density=True)
    ax.set_title(col.replace("_", " ").title(), fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlabel(col)
plt.suptitle("Vital Sign Distributions by Triage Acuity", fontsize=16, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig3_vitals_by_acuity.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code('''
# -- 6d: Chief complaint system distribution --
if "chief_complaint_system" in train_raw.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    ct = pd.crosstab(train_raw["chief_complaint_system"],
                     train_raw["triage_acuity"], normalize="index")
    ct.plot(kind="barh", stacked=True, ax=ax, color=PALETTE, edgecolor="white")
    ax.set_xlabel("Proportion", fontweight="bold")
    ax.set_title("ESI Acuity Distribution by Chief Complaint System", fontweight="bold")
    ax.legend(title="ESI", labels=[f"ESI-{i}" for i in range(1, 6)])
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig4_complaint_system.png", dpi=150, bbox_inches="tight")
    plt.show()
''')

code('''
# -- 6e: Comorbidity analysis --
hx_cols = [c for c in history.columns if c.startswith("hx_")]
train_with_hx = train_raw.merge(history, on="patient_id", how="left")
train_with_hx["total_hx"] = train_with_hx[hx_cols].sum(axis=1)
print("Mean comorbidity count by acuity:")
for acuity, val in train_with_hx.groupby("triage_acuity")["total_hx"].mean().items():
    print(f"  ESI-{acuity}: {val:.2f}")

fig, ax = plt.subplots(figsize=(14, 8))
hx_by_acuity = train_with_hx.groupby("triage_acuity")[hx_cols].mean()
hx_by_acuity.T.plot(kind="barh", ax=ax, color=PALETTE, edgecolor="white")
ax.set_xlabel("Prevalence Rate", fontweight="bold")
ax.set_title("Comorbidity Prevalence by ESI Acuity Level", fontweight="bold")
ax.legend(title="ESI", labels=[f"ESI-{i}" for i in range(1, 6)])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig5_comorbidities.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code('''
# -- 6f: Arrival patterns --
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
hourly = pd.crosstab(train_raw["arrival_hour"], train_raw["triage_acuity"], normalize="index")
hourly.plot(kind="area", stacked=True, ax=axes[0], color=PALETTE, alpha=0.8)
axes[0].set_title("Acuity Mix by Arrival Hour", fontweight="bold")
axes[0].set_xlabel("Hour of Day", fontweight="bold")
axes[0].set_ylabel("Proportion", fontweight="bold")
axes[0].legend(title="ESI", labels=[f"ESI-{i}" for i in range(1, 6)])

mode_acuity = pd.crosstab(train_raw["arrival_mode"], train_raw["triage_acuity"], normalize="index")
mode_acuity.plot(kind="barh", stacked=True, ax=axes[1], color=PALETTE, edgecolor="white")
axes[1].set_title("Acuity Mix by Arrival Mode", fontweight="bold")
axes[1].set_xlabel("Proportion", fontweight="bold")
axes[1].legend(title="ESI", labels=[f"ESI-{i}" for i in range(1, 6)])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig6_arrival_patterns.png", dpi=150, bbox_inches="tight")
plt.show()
''')

# --- Section 7: Feature Engineering -----------------------------------

md("""
## 7. Feature Engineering

The pipeline produces 153 engineered columns and uses 146 modeling features after
dropping IDs, raw text, and outcome-derived columns (`disposition`, `ed_los_hours`,
`site_id`, `triage_nurse_id`).

Feature groups:

- **Raw physiology and intake context**: BP, HR, RR, temperature, SpO2, GCS, pain score,
  age, arrival mode, prior utilization.
- **Derived clinical features**: shock index, MAP, pulse pressure, hemodynamic /
  temperature / respiratory abnormality flags, age-adjusted HR abnormality, critical
  flag counts.
- **Informative missingness**: per-vital missing flags, documentation completeness,
  total missing-vital count.
- **Complaint NLP heuristics**: complaint length, keyword acuity counts, targeted
  high-risk complaint flags.
- **Comorbidity composites**: cardiovascular, metabolic, respiratory, mental health,
  immunocompromised burden summaries.
- **Temporal patterns**: cyclical hour/day/month encodings plus night/evening/weekend
  flags.
""")

code('''
train_df = engineer_features(train_raw.copy(), complaints, history, is_train=True)
test_df = engineer_features(test_raw.copy(), complaints, history, is_train=False)

category_maps = build_category_maps(train_df)
train_df = encode_categoricals(train_df, category_maps)
test_df = encode_categoricals(test_df, category_maps)

feature_cols = get_feature_columns(train_df)
missing_in_test = [c for c in feature_cols if c not in test_df.columns]
if missing_in_test:
    print(f"[!] Adding missing columns to test set: {missing_in_test}")
    for c in missing_in_test:
        test_df[c] = 0
feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

y = train_df["triage_acuity"].values
print(f"\\nFinal feature count: {len(feature_cols)}")
print(f"Training samples: {len(train_df)}")
print(f"Test samples:     {len(test_df)}")

groups = {
    "Vital signs (raw)": [c for c in feature_cols if c in [
        "systolic_bp", "diastolic_bp", "heart_rate", "respiratory_rate",
        "temperature_c", "spo2", "gcs_total", "pain_score", "weight_kg", "height_cm"]],
    "Clinical derived": [c for c in feature_cols if any(c.startswith(p) for p in [
        "shock_", "map_", "pulse_pressure", "spo2_", "gcs_", "hypo", "hyper",
        "tachy", "brady", "fever", "pain_", "pp_", "hr_age", "n_critical"])],
    "Missingness": [c for c in feature_cols if "missing" in c or c == "doc_completeness"],
    "NLP / Chief complaint": [c for c in feature_cols if c.startswith("cc_") or (
        c.startswith("n_") and "keyword" in c) or c in [
        "complaint_length", "complaint_word_count", "keyword_acuity_signal"]],
    "Comorbidity": [c for c in feature_cols if c.startswith("hx_") or c == "total_hx_flags"],
    "Temporal": [c for c in feature_cols if any(c.startswith(p) for p in [
        "hour_", "day_", "month_", "is_night", "is_evening", "is_weekend"])],
    "Demographics": [c for c in feature_cols if c in [
        "age", "sex", "arrival_mode", "arrival_hour", "arrival_day", "arrival_month"]],
}
print("\\nFeature groups:")
for name, cols in groups.items():
    if cols:
        print(f"  {name:<25} {len(cols):>3} features")
''')

# --- Section 8: Modeling Pipeline --------------------------------------

md("""
## 8. Modeling Pipeline

Three base learners with 5-fold stratified CV (all evaluated on honest out-of-fold
predictions) feed a multinomial logistic meta-learner. The stacked ensemble is then
passed through an entropy-triggered **conservative shift** that nudges uncertain
predictions one level toward higher acuity.

| Stage | Model | Notes |
|---|---|---|
| Baseline | LightGBM | Standard cross-entropy, no class weights |
| Safety-weighted | LightGBM | Higher sample weights for ESI-1/2/5 |
| Safety-weighted | XGBoost | Diversity for the stacking meta-learner |
| Stacking | Multinomial logistic | Own 5-fold OOF evaluation |
| Post-processing | Conservative shift | Entropy threshold swept on OOF meta probs |
""")

code('''
# -- 8a: Baseline LightGBM (standard cross-entropy, no safety weighting) --
print("-- 8a: Baseline LightGBM --")
baseline_models, baseline_oof, baseline_preds = train_lgbm_cv(
    train_df, y, feature_cols,
    n_folds=5,
    params=LGBM_BASE_PARAMS.copy(),
    use_sample_weights=False,
    model_name="Baseline LightGBM",
)
baseline_report = full_safety_report(y, baseline_preds, model_name="Baseline LightGBM")
print_safety_report(baseline_report)
''')

code('''
# -- 8b: Safety-Weighted LightGBM --
print("-- 8b: Safety-Weighted LightGBM --")
safety_lgbm_models, safety_lgbm_oof, safety_lgbm_preds = train_lgbm_cv(
    train_df, y, feature_cols,
    n_folds=5,
    params=LGBM_BASE_PARAMS.copy(),
    use_sample_weights=True,
    model_name="Safety-Weighted LightGBM",
)
safety_lgbm_report = full_safety_report(y, safety_lgbm_preds, model_name="Safety-Weighted LightGBM")
print_safety_report(safety_lgbm_report)
''')

code('''
# -- 8c: XGBoost (diversity for ensemble) --
print("-- 8c: Safety-Weighted XGBoost --")
xgb_models, xgb_oof, xgb_preds = train_xgb_cv(
    train_df, y, feature_cols,
    n_folds=5,
    use_sample_weights=True,
    model_name="Safety-Weighted XGBoost",
)
xgb_report = full_safety_report(y, xgb_preds, model_name="Safety-Weighted XGBoost")
print_safety_report(xgb_report)
''')

code('''
# -- 8d: Stacking meta-learner --
print("-- 8d: Stacking meta-learner --")
oof_dict = {
    "lgbm_baseline": baseline_oof,
    "lgbm_safety":   safety_lgbm_oof,
    "xgb_safety":    xgb_oof,
}
meta_model, meta_scaler, meta_oof_proba, meta_oof_labels = train_stacking_meta(oof_dict, y)
stacked_report = full_safety_report(y, meta_oof_labels, model_name="Stacked Ensemble")
print_safety_report(stacked_report)
''')

code('''
# -- 8e: Uncertainty-aware conservative shift --
print("-- 8e: Conservative shift threshold sweep --")
best_threshold = 0.7
best_cwe = float("inf")
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    shifted = conservative_shift(meta_oof_proba, entropy_threshold=threshold)
    cwe = cost_weighted_error(y, shifted)
    utr = undertriage_rate(y, shifted)
    acc = (shifted == y).mean()
    print(f"  threshold={threshold:.1f}  Acc={acc:.4f}  UTR={utr:.4f}  CWE={cwe:.4f}")
    if cwe < best_cwe:
        best_cwe = cwe
        best_threshold = threshold

print(f"\\n  -> Best entropy threshold: {best_threshold}")
final_oof_preds = conservative_shift(meta_oof_proba, entropy_threshold=best_threshold)
final_report = full_safety_report(y, final_oof_preds, model_name="SafeTriageNet (Final)")
print_safety_report(final_report)
''')

# --- Section 9: Evaluation ---------------------------------------------

md("""
## 9. Evaluation and Clinical Safety Analysis

We compare five configurations side-by-side on accuracy, macro-F1, under-triage rate,
severe under-triage rate, over-triage rate, and cost-weighted error. The safety-aware
models trade a negligible amount of raw accuracy for a materially lower under-triage
rate — which is what we care about clinically.
""")

code('''
all_reports = [baseline_report, safety_lgbm_report, xgb_report, stacked_report, final_report]
comparison_df = pd.DataFrame(all_reports)
comparison_cols = [
    "model_name", "accuracy", "macro_f1", "weighted_f1",
    "quadratic_kappa", "undertriage_rate", "severe_undertriage_rate",
    "overtriage_rate", "cost_weighted_error", "safety_adjusted_kappa",
]
print(comparison_df[comparison_cols].to_string(index=False))
comparison_df[comparison_cols].to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
''')

code('''
# -- 9a: Confusion matrix comparison --
fig, axes = plt.subplots(1, 3, figsize=(22, 6))
for ax, (preds, name) in zip(axes, [
    (baseline_preds, "Baseline LightGBM"),
    (meta_oof_labels, "Stacked Ensemble"),
    (final_oof_preds, "SafeTriageNet (Final)"),
]):
    cm = confusion_matrix(y, preds, labels=[1, 2, 3, 4, 5])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
        xticklabels=[f"ESI-{i}" for i in range(1, 6)],
        yticklabels=[f"ESI-{i}" for i in range(1, 6)],
        linewidths=1, linecolor="white",
        cbar_kws={"label": "Proportion"},
    )
    ax.set_xlabel("Predicted", fontweight="bold")
    ax.set_ylabel("Actual", fontweight="bold")
    ax.set_title(name, fontweight="bold")
plt.suptitle("Confusion Matrices: Baseline vs Stacked vs Safety-Aware",
             fontsize=15, fontweight="bold", y=1.03)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig7_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code('''
# -- 9b: Safety / accuracy tradeoff --
fig, ax = plt.subplots(figsize=(10, 7))
for report, marker, size in zip(
    all_reports,
    ["o", "s", "D", "^", "*"],
    [100, 100, 100, 120, 200],
):
    ax.scatter(
        report["accuracy"], report["undertriage_rate"],
        s=size, marker=marker, label=report["model_name"],
        edgecolors="black", linewidth=1, zorder=5,
    )
ax.axhline(y=0.05, color="red", linestyle="--", alpha=0.7, label="ACS 5% UTR Threshold")
ax.set_xlabel("Accuracy", fontweight="bold", fontsize=13)
ax.set_ylabel("Under-Triage Rate (ESI 1-2 -> ESI 3-5)", fontweight="bold", fontsize=13)
ax.set_title("Safety-Accuracy Tradeoff Frontier\\nLower Under-Triage Rate = Safer",
             fontweight="bold", fontsize=14)
ax.legend(loc="upper right", fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig8_safety_accuracy_tradeoff.png", dpi=150, bbox_inches="tight")
plt.show()
''')

code('''
# -- 9c: Feature importance --
importance_df = get_feature_importance(safety_lgbm_models, feature_cols, importance_type="gain")
print("Top 30 features by mean gain:")
print(importance_df.head(30).to_string(index=False))

top_n = 30
top_features = importance_df.head(top_n).reset_index(drop=True)


def _feature_color(name: str) -> str:
    if "missing" in name or name == "doc_completeness":
        return "#e74c3c"       # red  -- missingness
    if (name.startswith("cc_")
            or "keyword" in name
            or name in {"complaint_length", "complaint_word_count", "keyword_acuity_signal"}):
        return "#2ecc71"       # green -- NLP
    if name.startswith("hx_"):
        return "#9b59b6"       # purple -- comorbidity
    return "#3498db"           # blue  -- other structured


feature_colors = [_feature_color(f) for f in top_features["feature"]]
y_positions = list(range(top_n - 1, -1, -1))  # top feature first

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(
    y_positions, top_features["mean_importance"].values,
    xerr=top_features["std_importance"].values,
    color=feature_colors, edgecolor="white", alpha=0.9, capsize=3,
)
ax.set_yticks(y_positions)
ax.set_yticklabels(top_features["feature"].values, fontsize=9)
ax.set_xlabel("Mean Gain (Feature Importance)", fontweight="bold")
ax.set_title(
    "Top 30 Features -- Safety-Weighted LightGBM\\n"
    "(red: missingness, green: NLP/complaint, purple: comorbidity, blue: other)",
    fontweight="bold",
)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "fig9_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()

importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
''')

code('''
# -- 9d: Fairness audit by sex and age group --
print("\\nFairness audit:")
for group_col, source_series in [
    ("Sex", train_raw["sex"]),
    ("Age Group", train_raw["age_group"]),
]:
    print(f"\\n  Performance by {group_col}:")
    for group in sorted(source_series.dropna().unique()):
        mask = source_series.values == group
        if mask.sum() < 50:
            continue
        group_true = y[mask]
        group_pred = final_oof_preds[mask]
        acc = (group_true == group_pred).mean()
        utr_ = undertriage_rate(group_true, group_pred)
        cwe_ = cost_weighted_error(group_true, group_pred)
        print(f"    {str(group):<20} n={mask.sum():>6,}  "
              f"Acc={acc:.4f}  UTR={utr_:.4f}  CWE={cwe_:.4f}")
''')

# --- Section 10: SHAP --------------------------------------------------

md("""
## 10. SHAP Explainability

Global (all-class) and ESI-1-specific SHAP summaries for the safety-weighted LightGBM.
SHAP is optional — if the package is unavailable, the block is skipped without failing
the rest of the notebook.
""")

code('''
try:
    import shap

    n_shap_samples = min(2000, len(train_df))
    rng_shap = np.random.default_rng(SEED)
    shap_idx = rng_shap.choice(len(train_df), size=n_shap_samples, replace=False)
    X_shap = train_df.iloc[shap_idx][feature_cols]

    explainer = shap.TreeExplainer(safety_lgbm_models[0])
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_per_class = shap_values
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_per_class = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
        mean_abs_shap = np.mean(np.abs(shap_values), axis=2)
    else:
        shap_per_class = None
        mean_abs_shap = np.abs(shap_values)

    fig, ax = plt.subplots(figsize=(14, 10))
    feature_shap_means = mean_abs_shap.mean(axis=0)
    top_idx = np.argsort(feature_shap_means)[-20:]
    shap_df = pd.DataFrame({
        "feature": [feature_cols[i] for i in top_idx],
        "mean_abs_shap": feature_shap_means[top_idx],
    }).sort_values("mean_abs_shap", ascending=True)
    ax.barh(range(len(shap_df)), shap_df["mean_abs_shap"].values,
            color="steelblue", edgecolor="white", alpha=0.9)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df["feature"].values)
    ax.set_xlabel("Mean |SHAP Value|", fontweight="bold")
    ax.set_title("SHAP Feature Importance -- Top 20 Features\\n"
                 "(Mean absolute SHAP across all ESI classes)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_shap_importance.png", dpi=150, bbox_inches="tight")
    plt.show()

    if shap_per_class is not None and len(shap_per_class) >= 1:
        fig, ax = plt.subplots(figsize=(14, 10))
        esi1_shap = np.abs(shap_per_class[0]).mean(axis=0)
        top_esi1 = np.argsort(esi1_shap)[-15:]
        esi1_df = pd.DataFrame({
            "feature": [feature_cols[i] for i in top_esi1],
            "mean_abs_shap": esi1_shap[top_esi1],
        }).sort_values("mean_abs_shap", ascending=True)
        ax.barh(range(len(esi1_df)), esi1_df["mean_abs_shap"].values,
                color="#e74c3c", edgecolor="white", alpha=0.9)
        ax.set_yticks(range(len(esi1_df)))
        ax.set_yticklabels(esi1_df["feature"].values)
        ax.set_xlabel("Mean |SHAP Value|", fontweight="bold")
        ax.set_title("SHAP Feature Importance -- ESI-1 (Resuscitation)\\n"
                     "What drives the highest-acuity predictions?", fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "fig11_shap_esi1.png", dpi=150, bbox_inches="tight")
        plt.show()
except Exception as exc:
    print(f"SHAP analysis skipped: {exc}")
''')

# --- Section 11: Clinical insights ------------------------------------

md("""
## 11. Clinical Insights and Limitations

Three findings stand out:

1. **Informative missingness is real but partial.** Four of six core vitals show
   significant acuity-linked missingness. Documentation completeness is one of the
   strongest features. `heart_rate` and `spo2` are not missing in this dataset — we
   therefore describe missingness as a useful but partial signal, not a universal rule.
2. **Safety vs accuracy trade-off is what the project buys.** Overall accuracy is
   essentially flat across all five model configurations. What changes is *which*
   mistakes are made — the final system reduces under-triage of ESI-1/2 patients
   relative to the accuracy-optimized baseline.
3. **A broad multimodal feature set beats a complicated modeling stack.** GCS, NEWS2,
   pain score, and SpO2 dominate, followed by complaint-text heuristics and comorbidity
   composites. We did not need a heavyweight language model to benefit from text.

**Limitations.**

1. Synthetic data. Real ED data is noisier and more institution-specific.
2. Lightweight text features only. Real triage notes contain abbreviations,
   misspellings, and clinical shorthand this pipeline does not model.
3. The conservative-shift threshold is selected from the same out-of-fold predictions
   used for evaluation — acceptable for a competition notebook, but not a fully nested
   estimate.
4. No probability calibration (Platt / isotonic). Clinical decision support requires
   calibrated confidence scores before any real deployment.
5. No prospective validation. Any downstream clinical use would require testing on real
   ED data and, in practice, a prospective study against current nurse triage.
""")

# --- Section 12: Submission --------------------------------------------

md("""
## 12. Generate Submission

Predictions come from the full stacking pipeline (averaged base-model predictions
over the five folds, then meta-learner, then conservative shift at the selected
entropy threshold). The submission file is written to the working directory.
""")

code('''
print("Generating test predictions...")
models_dict = {
    "lgbm_baseline": baseline_models,
    "lgbm_safety":   safety_lgbm_models,
    "xgb_safety":    xgb_models,
}
test_proba, _ = predict_stacked(
    models_dict,
    test_df,
    feature_cols,
    meta_model,
    meta_scaler,
)
test_final_preds = conservative_shift(test_proba, entropy_threshold=best_threshold)

submission = pd.DataFrame({
    "patient_id": test_df["patient_id"],
    "triage_acuity": test_final_preds.astype(int),
})

submission_path = OUTPUT_DIR / "submission.csv"
submission.to_csv(submission_path, index=False)

print(f"\\nSubmission saved to: {submission_path}")
print(f"Submission shape:    {submission.shape}")
print("\\nPrediction distribution:")
print(submission["triage_acuity"].value_counts().sort_index().to_string())

assert list(submission.columns) == list(sample_sub.columns), "Submission format mismatch!"
assert len(submission) == len(sample_sub), "Submission length mismatch!"
print("\\n[OK] Submission format verified.")

print(f"\\nFinal SafeTriageNet metrics (out-of-fold):")
print(f"  Accuracy:      {final_report['accuracy']:.4f}")
print(f"  Macro F1:      {final_report['macro_f1']:.4f}")
print(f"  Under-triage:  {final_report['undertriage_rate']:.4f}")
print(f"  CWE:           {final_report['cost_weighted_error']:.4f}")
''')

# --- Build ------------------------------------------------------------

def build() -> None:
    nb = new_notebook()
    nb.cells = CELLS
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.10"},
    }
    OUT_NB.parent.mkdir(parents=True, exist_ok=True)
    with OUT_NB.open("w", encoding="utf-8") as fp:
        nbformat.write(nb, fp)
    print(f"Wrote {OUT_NB} ({len(nb.cells)} cells)")


if __name__ == "__main__":
    build()
