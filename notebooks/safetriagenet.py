#!/usr/bin/env python
# coding: utf-8
"""
+==============================================================================+
|                                                                              |
|    SafeTriageNet: Safety-Aware Multimodal Triage with Informative            |
|    Missingness and Asymmetric Clinical Cost                                  |
|                                                                              |
|    Triagegeist Competition -- Laitinen-Fredriksson Foundation                 |
|                                                                              |
|    "When Getting It Wrong Matters More Than Getting It Right"                |
|                                                                              |
+==============================================================================+

This notebook implements a safety-aware multimodal triage decision support
system. Unlike standard classification approaches that optimize only for
accuracy, SafeTriageNet prioritizes clinically dangerous errors through
asymmetric evaluation, class weighting, and conservative post-processing.

Three pillars:
    1. Informative Missingness -- treating missing vitals as a clinical signal
    2. Multimodal Fusion -- structured vitals + NLP on chief complaints + comorbidities
    3. Safety-Aware Optimization -- asymmetric cost, conservative uncertainty shifting

Author: SafeTriageNet Team
License: MIT
"""

# ===============================================================================
# 1. SETUP & IMPORTS
# ===============================================================================

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Non-interactive backend for headless execution
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import lightgbm as lgb
import xgboost as xgb

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from features import (
    engineer_features, get_feature_columns, build_category_maps, encode_categoricals,
    CORE_VITALS, VITAL_COLUMNS
)
from safety import (
    full_safety_report, print_safety_report, cost_weighted_error,
    undertriage_rate, overtriage_rate, severe_undertriage_rate,
    conservative_shift, get_cost_matrix, CLINICAL_COST_MATRIX
)
from models import (
    train_lgbm_cv, train_xgb_cv, train_stacking_meta,
    predict_stacked, get_feature_importance, LGBM_BASE_PARAMS
)

# -- Plot styling --
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'figure.figsize': (14, 8),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'figure.dpi': 100,
})
PALETTE = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db']
ESI_NAMES = {1: 'Resuscitation', 2: 'Emergent', 3: 'Urgent', 4: 'Less Urgent', 5: 'Non-urgent'}

# -- Random seed --
SEED = 42
np.random.seed(SEED)

print("SafeTriageNet -- Setup complete.\n")


REQUIRED_DATA_FILES = [
    'train.csv',
    'test.csv',
    'chief_complaints.csv',
    'patient_history.csv',
    'sample_submission.csv',
]


def resolve_data_dir() -> str:
    """Locate the competition data directory from a small set of reproducible paths."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []

    env_data_dir = os.environ.get('TRIAGEGEIST_DATA_DIR')
    if env_data_dir:
        candidates.append(env_data_dir)

    candidates.extend([
        os.path.join(here, '..', '..', 'triagegeist'),
        os.path.join(here, '..', 'triagegeist'),
        os.path.join(here, '..', 'data', 'triagegeist'),
    ])

    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if all(os.path.exists(os.path.join(candidate, filename)) for filename in REQUIRED_DATA_FILES):
            return candidate

    raise FileNotFoundError(
        "Could not locate the Triagegeist data directory. "
        "Set TRIAGEGEIST_DATA_DIR or place the CSV files in one of: "
        + ", ".join(os.path.abspath(path) for path in candidates if path)
    )


# ===============================================================================
# 2. DATA LOADING
# ===============================================================================

print("=" * 70)
print("  SECTION 2: DATA LOADING")
print("=" * 70)

DATA_DIR = resolve_data_dir()

train_raw = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
test_raw = pd.read_csv(os.path.join(DATA_DIR, 'test.csv'))
complaints = pd.read_csv(os.path.join(DATA_DIR, 'chief_complaints.csv'))
history = pd.read_csv(os.path.join(DATA_DIR, 'patient_history.csv'))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, 'sample_submission.csv'))

print(f"\n  Train set:         {train_raw.shape[0]:>7,} records x {train_raw.shape[1]} columns")
print(f"  Test set:          {test_raw.shape[0]:>7,} records x {test_raw.shape[1]} columns")
print(f"  Chief complaints:  {complaints.shape[0]:>7,} records")
print(f"  Patient history:   {history.shape[0]:>7,} records")
print(f"  Sample submission: {sample_sub.shape[0]:>7,} records")
print(f"  Data directory:    {DATA_DIR}")

print(f"\n  Train columns: {list(train_raw.columns)}")
print(f"\n  Target distribution (triage_acuity):")
print(train_raw['triage_acuity'].value_counts().sort_index().to_string())


# ===============================================================================
# 3. EXPLORATORY DATA ANALYSIS
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SECTION 3: EXPLORATORY DATA ANALYSIS")
print("=" * 70)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -- 3a: Target Distribution --
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Raw counts
counts = train_raw['triage_acuity'].value_counts().sort_index()
colors = [PALETTE[i] for i in range(5)]
bars = axes[0].bar(counts.index, counts.values, color=colors, edgecolor='white', linewidth=1.5)
axes[0].set_xlabel('ESI Acuity Level', fontweight='bold')
axes[0].set_ylabel('Patient Count', fontweight='bold')
axes[0].set_title('Triage Acuity Distribution (Training Set)')
for bar, count in zip(bars, counts.values):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 200,
                 f'{count:,}\n({count/len(train_raw)*100:.1f}%)',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')

# ESI labels
for i, (esi, name) in enumerate(ESI_NAMES.items()):
    axes[0].text(esi, -max(counts.values)*0.06, name, ha='center', fontsize=8,
                 style='italic', color='gray')

# Clinical cost matrix heatmap
im = axes[1].imshow(CLINICAL_COST_MATRIX, cmap='YlOrRd', aspect='auto')
axes[1].set_xticks(range(5))
axes[1].set_yticks(range(5))
axes[1].set_xticklabels([f'ESI-{i+1}' for i in range(5)])
axes[1].set_yticklabels([f'ESI-{i+1}' for i in range(5)])
axes[1].set_xlabel('Predicted Acuity', fontweight='bold')
axes[1].set_ylabel('Actual Acuity', fontweight='bold')
axes[1].set_title('Asymmetric Clinical Cost Matrix')
for i in range(5):
    for j in range(5):
        color = 'white' if CLINICAL_COST_MATRIX[i, j] > 8 else 'black'
        axes[1].text(j, i, f'{CLINICAL_COST_MATRIX[i, j]:.1f}',
                     ha='center', va='center', fontsize=12, fontweight='bold', color=color)
plt.colorbar(im, ax=axes[1], label='Clinical Cost')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_target_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 1: Target distribution & cost matrix saved")


# -- 3b: Missingness Analysis (CORE INSIGHT) --
print("\n-- 3b: INFORMATIVE MISSINGNESS ANALYSIS --")
print("  The dataset states: 'Missingness is not random and reflects realistic")
print("  triage conditions in which certain vitals are not obtained for")
print("  lower-acuity patients.'")
print("  Let's test this hypothesis quantitatively.\n")

# Compute missingness rates by acuity level
vital_cols_present = [c for c in CORE_VITALS if c in train_raw.columns]
miss_by_acuity = pd.DataFrame()
for col in vital_cols_present:
    for acuity in sorted(train_raw['triage_acuity'].unique()):
        subset = train_raw[train_raw['triage_acuity'] == acuity]
        miss_rate = subset[col].isna().mean()
        miss_by_acuity = pd.concat([miss_by_acuity, pd.DataFrame({
            'vital': [col], 'acuity': [acuity], 'missing_rate': [miss_rate]
        })], ignore_index=True)

# Also check pain_score (-1 encoding)
if 'pain_score' in train_raw.columns:
    for acuity in sorted(train_raw['triage_acuity'].unique()):
        subset = train_raw[train_raw['triage_acuity'] == acuity]
        miss_rate = (subset['pain_score'] == -1).mean()
        miss_by_acuity = pd.concat([miss_by_acuity, pd.DataFrame({
            'vital': ['pain_score'], 'acuity': [acuity], 'missing_rate': [miss_rate]
        })], ignore_index=True)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# Heatmap: missingness rate by vital x acuity
pivot = miss_by_acuity.pivot(index='vital', columns='acuity', values='missing_rate')
sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[0],
            linewidths=1, linecolor='white',
            xticklabels=[f'ESI-{i}' for i in pivot.columns],
            cbar_kws={'label': 'Missing Rate'})
axes[0].set_title('Vital Sign Missing Rate by ESI Acuity Level\n(Higher = More Missing)',
                   fontweight='bold')
axes[0].set_xlabel('Triage Acuity (ESI)', fontweight='bold')
axes[0].set_ylabel('')

# Documentation completeness by acuity
n_vitals_check = train_raw[vital_cols_present].notna().sum(axis=1)
train_raw['_temp_doc_completeness'] = n_vitals_check / len(vital_cols_present)
doc_comp_by_acuity = train_raw.groupby('triage_acuity')['_temp_doc_completeness'].mean()
bars = axes[1].bar(doc_comp_by_acuity.index, doc_comp_by_acuity.values,
                    color=colors, edgecolor='white', linewidth=1.5)
axes[1].set_xlabel('ESI Acuity Level', fontweight='bold')
axes[1].set_ylabel('Mean Documentation Completeness', fontweight='bold')
axes[1].set_title('Vital Sign Documentation Completeness by Acuity\n(Higher = More Vitals Recorded)',
                   fontweight='bold')
axes[1].set_ylim(0, 1.05)
for bar, val in zip(bars, doc_comp_by_acuity.values):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
train_raw.drop('_temp_doc_completeness', axis=1, inplace=True)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_missingness_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 2: Missingness analysis saved")

# Statistical test: Chi-square test for association between missingness and acuity
print("\n  -- Chi-Square Test: Missingness x Acuity --")
significant_missingness = []
non_significant_missingness = []
for col in vital_cols_present:
    missing_flag = train_raw[col].isna().astype(int)
    contingency = pd.crosstab(missing_flag, train_raw['triage_acuity'])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
    print(f"    {col:<22} chi2={chi2:>10.1f}  p={p_val:.2e}  {sig}")
    if p_val < 0.05:
        significant_missingness.append(col)
    else:
        non_significant_missingness.append(col)

print(
    f"\n  -> Conclusion: {len(significant_missingness)}/{len(vital_cols_present)} core vitals "
    "show statistically significant acuity-linked missingness."
)
if non_significant_missingness:
    print(f"     Non-significant in this dataset: {', '.join(non_significant_missingness)}")


# -- 3c: Vital Signs by Acuity --
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
vital_plot_cols = ['heart_rate', 'systolic_bp', 'respiratory_rate',
                   'spo2', 'temperature_c', 'gcs_total']

for idx, col in enumerate(vital_plot_cols):
    ax = axes[idx // 3][idx % 3]
    if col in train_raw.columns:
        for acuity in [1, 2, 3, 4, 5]:
            data = train_raw[train_raw['triage_acuity'] == acuity][col].dropna()
            if len(data) > 0:
                ax.hist(data, bins=40, alpha=0.4, label=f'ESI-{acuity}',
                       color=PALETTE[acuity-1], density=True)
        ax.set_title(col.replace('_', ' ').title(), fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xlabel(col)

plt.suptitle('Vital Sign Distributions by Triage Acuity', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_vitals_by_acuity.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 3: Vital distributions by acuity saved")


# -- 3d: Chief Complaint Analysis --
print("\n-- 3d: CHIEF COMPLAINT ANALYSIS --")
train_with_cc = train_raw.merge(complaints[['patient_id', 'chief_complaint_raw']],
                                 on='patient_id', how='left')

# Complaint length by acuity
train_with_cc['cc_length'] = train_with_cc['chief_complaint_raw'].fillna('').str.len()
cc_len_by_acuity = train_with_cc.groupby('triage_acuity')['cc_length'].describe()
print("\n  Chief Complaint Length by Acuity:")
print(cc_len_by_acuity[['mean', 'std', 'min', 'max']].round(1).to_string())

# Chief complaint system distribution
if 'chief_complaint_system' in train_raw.columns:
    fig, ax = plt.subplots(figsize=(14, 8))
    ct = pd.crosstab(train_raw['chief_complaint_system'],
                      train_raw['triage_acuity'], normalize='index')
    ct.plot(kind='barh', stacked=True, ax=ax, color=PALETTE, edgecolor='white')
    ax.set_xlabel('Proportion', fontweight='bold')
    ax.set_title('ESI Acuity Distribution by Chief Complaint System', fontweight='bold')
    ax.legend(title='ESI', labels=[f'ESI-{i}' for i in range(1,6)])
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_complaint_system.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Figure 4: Complaint system distribution saved")


# -- 3e: Patient History / Comorbidity Analysis --
print("\n-- 3e: COMORBIDITY ANALYSIS --")
hx_cols = [c for c in history.columns if c.startswith('hx_')]
train_with_hx = train_raw.merge(history, on='patient_id', how='left')

# Mean comorbidity count by acuity
train_with_hx['total_hx'] = train_with_hx[hx_cols].sum(axis=1)
hx_by_acuity = train_with_hx.groupby('triage_acuity')['total_hx'].mean()
print("\n  Mean comorbidity count by acuity:")
for acuity, val in hx_by_acuity.items():
    print(f"    ESI-{acuity}: {val:.2f}")

# Top comorbidities by acuity
fig, ax = plt.subplots(figsize=(14, 8))
hx_by_acuity_df = train_with_hx.groupby('triage_acuity')[hx_cols].mean()
hx_by_acuity_df.T.plot(kind='barh', ax=ax, color=PALETTE, edgecolor='white')
ax.set_xlabel('Prevalence Rate', fontweight='bold')
ax.set_title('Comorbidity Prevalence by ESI Acuity Level', fontweight='bold')
ax.legend(title='ESI', labels=[f'ESI-{i}' for i in range(1,6)])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_comorbidities.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 5: Comorbidity analysis saved")


# -- 3f: Arrival Patterns --
print("\n-- 3f: ARRIVAL PATTERNS --")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# By hour
hourly = pd.crosstab(train_raw['arrival_hour'], train_raw['triage_acuity'], normalize='index')
hourly.plot(kind='area', stacked=True, ax=axes[0], color=PALETTE, alpha=0.8)
axes[0].set_title('Acuity Mix by Arrival Hour', fontweight='bold')
axes[0].set_xlabel('Hour of Day', fontweight='bold')
axes[0].set_ylabel('Proportion', fontweight='bold')
axes[0].legend(title='ESI', labels=[f'ESI-{i}' for i in range(1,6)])

# By arrival mode
mode_acuity = pd.crosstab(train_raw['arrival_mode'], train_raw['triage_acuity'], normalize='index')
mode_acuity.plot(kind='barh', stacked=True, ax=axes[1], color=PALETTE, edgecolor='white')
axes[1].set_title('Acuity Mix by Arrival Mode', fontweight='bold')
axes[1].set_xlabel('Proportion', fontweight='bold')
axes[1].legend(title='ESI', labels=[f'ESI-{i}' for i in range(1,6)])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_arrival_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 6: Arrival patterns saved")


# ===============================================================================
# 4. FEATURE ENGINEERING
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SECTION 4: FEATURE ENGINEERING")
print("=" * 70)

# Apply full feature engineering pipeline
train_df = engineer_features(train_raw.copy(), complaints, history, is_train=True)
test_df = engineer_features(test_raw.copy(), complaints, history, is_train=False)

# Encode categoricals
category_maps = build_category_maps(train_df)
train_df = encode_categoricals(train_df, category_maps)
test_df = encode_categoricals(test_df, category_maps)

# Get feature columns
feature_cols = get_feature_columns(train_df)

# Ensure test has the same columns
missing_in_test = [c for c in feature_cols if c not in test_df.columns]
if missing_in_test:
    print(f"\n  [!] Adding missing columns to test set: {missing_in_test}")
    for c in missing_in_test:
        test_df[c] = 0

# Ensure feature_cols only contains columns present in both
feature_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

y = train_df['triage_acuity'].values

print(f"\n  Final feature count: {len(feature_cols)}")
print(f"  Training samples: {len(train_df)}")
print(f"  Test samples: {len(test_df)}")
print(f"\n  Feature groups:")

# Count features by group
groups = {
    'Vital signs (raw)': [c for c in feature_cols if c in ['systolic_bp', 'diastolic_bp', 'heart_rate', 'respiratory_rate', 'temperature_c', 'spo2', 'gcs_total', 'pain_score', 'weight_kg', 'height_cm']],
    'Clinical derived': [c for c in feature_cols if any(c.startswith(p) for p in ['shock_', 'map_', 'pulse_pressure', 'spo2_', 'gcs_', 'hypo', 'hyper', 'tachy', 'brady', 'fever', 'pain_', 'pp_', 'hr_age', 'n_critical'])],
    'Missingness': [c for c in feature_cols if 'missing' in c or c == 'doc_completeness'],
    'NLP / Chief complaint': [c for c in feature_cols if c.startswith('cc_') or c.startswith('n_') and 'keyword' in c or c in ['complaint_length', 'complaint_word_count', 'keyword_acuity_signal']],
    'Comorbidity': [c for c in feature_cols if c.startswith('hx_') or c == 'total_hx_flags'],
    'Temporal': [c for c in feature_cols if any(c.startswith(p) for p in ['hour_', 'day_', 'month_', 'is_night', 'is_evening', 'is_weekend'])],
    'Demographics': [c for c in feature_cols if c in ['age', 'sex', 'arrival_mode', 'arrival_hour', 'arrival_day', 'arrival_month']],
}
for group_name, cols in groups.items():
    if cols:
        print(f"    {group_name:<25} {len(cols):>3} features")


# ===============================================================================
# 5. MODELING PIPELINE
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SECTION 5: MODELING PIPELINE")
print("=" * 70)

# -- 5a: Baseline LightGBM (standard cross-entropy, no safety weighting) --
print("\n-- 5a: BASELINE MODEL (Standard LightGBM, no safety weighting) --")
baseline_params = LGBM_BASE_PARAMS.copy()

baseline_models, baseline_oof, baseline_preds = train_lgbm_cv(
    train_df, y, feature_cols,
    n_folds=5,
    params=baseline_params,
    use_sample_weights=False,
    model_name="Baseline LightGBM"
)

baseline_report = full_safety_report(y, baseline_preds, model_name="Baseline LightGBM")
print_safety_report(baseline_report)


# -- 5b: Safety-Weighted LightGBM --
print("\n-- 5b: SAFETY-WEIGHTED MODEL (LightGBM + acuity sample weights) --")
safety_lgbm_models, safety_lgbm_oof, safety_lgbm_preds = train_lgbm_cv(
    train_df, y, feature_cols,
    n_folds=5,
    params=LGBM_BASE_PARAMS.copy(),
    use_sample_weights=True,
    model_name="Safety-Weighted LightGBM"
)

safety_lgbm_report = full_safety_report(y, safety_lgbm_preds, model_name="Safety-Weighted LightGBM")
print_safety_report(safety_lgbm_report)


# -- 5c: XGBoost (diversity for ensemble) --
print("\n-- 5c: XGBOOST MODEL (diversity for stacking) --")
xgb_models, xgb_oof, xgb_preds = train_xgb_cv(
    train_df, y, feature_cols,
    n_folds=5,
    use_sample_weights=True,
    model_name="Safety-Weighted XGBoost"
)

xgb_report = full_safety_report(y, xgb_preds, model_name="Safety-Weighted XGBoost")
print_safety_report(xgb_report)


# -- 5d: Stacking Ensemble --
print("\n-- 5d: STACKING ENSEMBLE --")
oof_dict = {
    'lgbm_baseline': baseline_oof,
    'lgbm_safety': safety_lgbm_oof,
    'xgb_safety': xgb_oof,
}

meta_model, meta_scaler, meta_oof_proba, meta_oof_labels = train_stacking_meta(oof_dict, y)

stacked_report = full_safety_report(y, meta_oof_labels, model_name="Stacked Ensemble")
print_safety_report(stacked_report)


# -- 5e: Safety-Aware Conservative Shifting --
print("\n-- 5e: UNCERTAINTY-AWARE CONSERVATIVE SHIFTING --")
print("  Applying conservative shift: when prediction entropy is high,")
print("  err on the side of higher acuity (lower ESI number) for patient safety.\n")

# Try different entropy thresholds
best_threshold = 0.7
best_cwe = float('inf')

for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    shifted_preds = conservative_shift(meta_oof_proba, entropy_threshold=threshold)
    cwe = cost_weighted_error(y, shifted_preds)
    utr = undertriage_rate(y, shifted_preds)
    acc = (shifted_preds == y).mean()
    print(f"  Threshold={threshold:.1f}  |  Accuracy={acc:.4f}  |  UTR={utr:.4f}  |  CWE={cwe:.4f}")
    
    if cwe < best_cwe:
        best_cwe = cwe
        best_threshold = threshold

print(f"\n  -> Best entropy threshold: {best_threshold}")

# Final predictions with conservative shifting
final_oof_preds = conservative_shift(meta_oof_proba, entropy_threshold=best_threshold)
final_report = full_safety_report(y, final_oof_preds, model_name="SafeTriageNet (Final)")
print_safety_report(final_report)


# ===============================================================================
# 6. EVALUATION & CLINICAL SAFETY ANALYSIS
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SECTION 6: EVALUATION & SAFETY ANALYSIS")
print("=" * 70)

# -- 6a: Head-to-Head Comparison --
print("\n-- 6a: HEAD-TO-HEAD MODEL COMPARISON --")

all_reports = [baseline_report, safety_lgbm_report, xgb_report, stacked_report, final_report]
comparison_df = pd.DataFrame(all_reports)
comparison_cols = ['model_name', 'accuracy', 'macro_f1', 'weighted_f1',
                   'quadratic_kappa', 'undertriage_rate', 'severe_undertriage_rate',
                   'overtriage_rate', 'cost_weighted_error', 'safety_adjusted_kappa']
print("\n" + comparison_df[comparison_cols].to_string(index=False))

# Save comparison table
comparison_df[comparison_cols].to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)


# -- 6b: Confusion Matrix Comparison --
fig, axes = plt.subplots(1, 3, figsize=(22, 6))

for ax, (preds, name) in zip(axes, [
    (baseline_preds, 'Baseline LightGBM'),
    (meta_oof_labels, 'Stacked Ensemble'),
    (final_oof_preds, 'SafeTriageNet (Final)')
]):
    cm = confusion_matrix(y, preds, labels=[1, 2, 3, 4, 5])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=[f'ESI-{i}' for i in range(1,6)],
                yticklabels=[f'ESI-{i}' for i in range(1,6)],
                linewidths=1, linecolor='white',
                cbar_kws={'label': 'Proportion'})
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(name, fontweight='bold')

plt.suptitle('Confusion Matrices: Baseline vs Stacked vs Safety-Aware',
             fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig7_confusion_matrices.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 7: Confusion matrix comparison saved")


# -- 6c: Safety-Accuracy Tradeoff Frontier --
fig, ax = plt.subplots(figsize=(10, 7))

for report, marker, size in zip(all_reports,
                                  ['o', 's', 'D', '^', '*'],
                                  [100, 100, 100, 120, 200]):
    ax.scatter(report['accuracy'], report['undertriage_rate'],
               s=size, marker=marker, label=report['model_name'],
               edgecolors='black', linewidth=1, zorder=5)

ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='ACS 5% UTR Threshold')
ax.set_xlabel('Accuracy', fontweight='bold', fontsize=13)
ax.set_ylabel('Under-Triage Rate (ESI 1-2 -> ESI 3-5)', fontweight='bold', fontsize=13)
ax.set_title('Safety-Accuracy Tradeoff Frontier\nLower Under-Triage Rate = Safer',
             fontweight='bold', fontsize=14)
ax.legend(loc='upper right', fontsize=10)
ax.invert_yaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig8_safety_accuracy_tradeoff.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 8: Safety-accuracy tradeoff saved")


# -- 6d: Feature Importance Analysis --
print("\n-- 6d: FEATURE IMPORTANCE (SHAP-like via LightGBM gain) --")

importance_df = get_feature_importance(safety_lgbm_models, feature_cols, importance_type='gain')
print(f"\n  Top 30 Features by Mean Gain:")
print(importance_df.head(30).to_string(index=False))

fig, ax = plt.subplots(figsize=(12, 10))
top_n = 30
top_features = importance_df.head(top_n)
bars = ax.barh(range(top_n-1, -1, -1), top_features['mean_importance'].values,
               xerr=top_features['std_importance'].values,
               color='steelblue', edgecolor='white', alpha=0.9, capsize=3)

# Color features by category. bars[i] corresponds to top_features.iloc[i]
# (both in descending-importance order), so index i -- not top_n-1-i.
for i, (_, row) in enumerate(top_features.iterrows()):
    if 'missing' in row['feature'] or row['feature'] == 'doc_completeness':
        bars[i].set_color('#e74c3c')
    elif row['feature'].startswith('cc_') or 'keyword' in row['feature'] or row['feature'] in ['complaint_length', 'complaint_word_count', 'keyword_acuity_signal']:
        bars[i].set_color('#2ecc71')
    elif row['feature'].startswith('hx_'):
        bars[i].set_color('#9b59b6')

ax.set_yticks(range(top_n-1, -1, -1))
ax.set_yticklabels(top_features['feature'].values, fontsize=9)
ax.set_xlabel('Mean Gain (Feature Importance)', fontweight='bold')
ax.set_title('Top 30 Features -- Safety-Weighted LightGBM\n'
             '((RED) Missingness  (GRN) NLP/Complaint  (PUR) Comorbidity  (BLU) Other)',
             fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig9_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  -> Figure 9: Feature importance saved")

# Save full importance ranking
importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance.csv'), index=False)


# -- 6e: Fairness Audit --
print("\n-- 6e: FAIRNESS AUDIT BY DEMOGRAPHICS --")

# Reconstruct sex from encoded values for analysis
sex_mapping = {
    v: k for k, v in zip(
        train_raw['sex'].astype('category').cat.categories,
        range(len(train_raw['sex'].astype('category').cat.categories))
    )
}

# Decode sex back for plotting
train_df['_sex_label'] = train_raw['sex'].values

for group_col, group_name in [('_sex_label', 'Sex'), ('age_group', 'Age Group')]:
    if group_col in train_df.columns or group_col == '_sex_label':
        print(f"\n  Performance by {group_name}:")
        
        source_col = group_col
        if group_col == 'age_group':
            train_df['_age_group_label'] = train_raw['age_group'].values
            source_col = '_age_group_label'
        
        groups_unique = train_df[source_col].unique()
        for group in sorted(groups_unique):
            mask = train_df[source_col] == group
            if mask.sum() < 50:
                continue
            group_true = y[mask]
            group_pred = final_oof_preds[mask]
            acc = (group_true == group_pred).mean()
            utr = undertriage_rate(group_true, group_pred)
            cwe_val = cost_weighted_error(group_true, group_pred)
            print(f"    {str(group):<20} n={mask.sum():>6,}  "
                  f"Acc={acc:.4f}  UTR={utr:.4f}  CWE={cwe_val:.4f}")

# Clean up temp columns
for col in ['_sex_label', '_age_group_label']:
    if col in train_df.columns:
        train_df.drop(col, axis=1, inplace=True)


# ===============================================================================
# 7. SHAP EXPLAINABILITY
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SECTION 7: SHAP EXPLAINABILITY ANALYSIS")
print("=" * 70)

try:
    import shap
    
    # Use a sample for SHAP (computational cost)
    n_shap_samples = min(2000, len(train_df))
    shap_idx = np.random.choice(len(train_df), n_shap_samples, replace=False)
    X_shap = train_df.iloc[shap_idx][feature_cols]
    
    # SHAP for the best LightGBM model (fold 0)
    explainer = shap.TreeExplainer(safety_lgbm_models[0])
    shap_values = explainer.shap_values(X_shap)
    
    # Handle both old format (list of arrays) and new format (3D ndarray)
    if isinstance(shap_values, list):
        # Old format: list of (n_samples, n_features) arrays, one per class
        shap_per_class = shap_values
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # New format: (n_samples, n_features, n_classes)
        shap_per_class = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
        mean_abs_shap = np.mean(np.abs(shap_values), axis=2)  # average over classes
    else:
        # Fallback: 2D array (binary or single output)
        shap_per_class = None
        mean_abs_shap = np.abs(shap_values)
    
    # Summary plot -- all classes
    fig, ax = plt.subplots(figsize=(14, 10))
    
    feature_shap_means = mean_abs_shap.mean(axis=0)
    top_shap_idx = np.argsort(feature_shap_means)[-20:]
    
    shap_df = pd.DataFrame({
        'feature': [feature_cols[i] for i in top_shap_idx],
        'mean_abs_shap': feature_shap_means[top_shap_idx]
    }).sort_values('mean_abs_shap', ascending=True)
    
    bars = ax.barh(range(len(shap_df)), shap_df['mean_abs_shap'].values,
                   color='steelblue', edgecolor='white', alpha=0.9)
    ax.set_yticks(range(len(shap_df)))
    ax.set_yticklabels(shap_df['feature'].values)
    ax.set_xlabel('Mean |SHAP Value|', fontweight='bold')
    ax.set_title('SHAP Feature Importance -- Top 20 Features\n'
                 '(Mean absolute SHAP across all ESI classes)', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig10_shap_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  -> Figure 10: SHAP importance saved")
    
    # Per-class SHAP for ESI-1 (most critical)
    if shap_per_class is not None and len(shap_per_class) >= 1:
        fig, ax = plt.subplots(figsize=(14, 10))
        esi1_shap = np.abs(shap_per_class[0]).mean(axis=0)
        top_esi1_idx = np.argsort(esi1_shap)[-15:]
        
        esi1_df = pd.DataFrame({
            'feature': [feature_cols[i] for i in top_esi1_idx],
            'mean_abs_shap': esi1_shap[top_esi1_idx]
        }).sort_values('mean_abs_shap', ascending=True)
        
        ax.barh(range(len(esi1_df)), esi1_df['mean_abs_shap'].values,
                color='#e74c3c', edgecolor='white', alpha=0.9)
        ax.set_yticks(range(len(esi1_df)))
        ax.set_yticklabels(esi1_df['feature'].values)
        ax.set_xlabel('Mean |SHAP Value|', fontweight='bold')
        ax.set_title('SHAP Feature Importance -- ESI-1 (Resuscitation)\n'
                     'What drives the highest-acuity predictions?', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'fig11_shap_esi1.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print("  -> Figure 11: SHAP ESI-1 importance saved")

except Exception as e:
    print(f"  [!] SHAP analysis skipped: {e}")


# ===============================================================================
# 8. CLINICAL INSIGHTS & KEY FINDINGS
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SECTION 8: KEY CLINICAL FINDINGS")
print("=" * 70)

print("""
  +---------------------------------------------------------------------+
  |  KEY FINDING 1: INFORMATIVE MISSINGNESS                            |
  |                                                                     |
  |  Missingness is informative, but not uniformly so across every      |
  |  vital. Documentation completeness is one of the strongest          |
  |  features in the model, and several vitals show clear acuity-linked |
  |  missingness patterns. In this dataset, absence of measurement is   |
  |  often a clinical signal rather than simple data quality noise.     |
  +---------------------------------------------------------------------+

  +---------------------------------------------------------------------+
  |  KEY FINDING 2: SAFETY vs ACCURACY TRADEOFF                        |
  |                                                                     |
  |  SafeTriageNet achieves a meaningful reduction in under-triage rate |
  |  (the most clinically dangerous error) compared to the accuracy-   |
  |  optimized baseline. Overall accuracy remains effectively flat,     |
  |  but the error profile shifts toward safer mistakes.                |
  +---------------------------------------------------------------------+

  +---------------------------------------------------------------------+
  |  KEY FINDING 3: MULTIMODAL FUSION MATTERS                          |
  |                                                                     |
  |  The feature set benefits from combining structured vitals, chief   |
  |  complaint text signals, and comorbidity history. On top of that,   |
  |  stacking complementary tree models improves under-triage behavior  |
  |  relative to any single base learner, even when overall accuracy    |
  |  remains nearly unchanged.                                          |
  +---------------------------------------------------------------------+
""")

# -- Clinical limitations (honest) --
print("""
  -- LIMITATIONS --
  
  1. SYNTHETIC DATA: All records are simulated. While distributions match
     published literature, real-world clinical relationships may be more 
     complex, noisy, and institution-specific.
     
  2. NO PROSPECTIVE VALIDATION: This is a retrospective analysis on synthetic
     data. Clinical deployment would require prospective validation with
     real patient outcomes.
     
  3. NLP ON SYNTHETIC TEXT: The chief complaint narratives are generated
     text -- real triage notes contain abbreviations, misspellings, and
     clinical shorthand that would require more robust NLP preprocessing.
     
  4. CALIBRATION: Probability calibration (Platt scaling, isotonic regression)
     was not applied in this version. Clinical decision support requires
     well-calibrated confidence scores.
     
  5. INSTITUTIONAL VARIANCE: Triage patterns vary dramatically between EDs.
     A model trained on one institution's data may not generalize.
""")


# ===============================================================================
# 9. GENERATE SUBMISSION
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SECTION 9: GENERATING SUBMISSION")
print("=" * 70)

# Generate test predictions using the full stacking pipeline
print("\n  Generating test predictions...")

models_dict = {
    'lgbm_baseline': baseline_models,
    'lgbm_safety': safety_lgbm_models,
    'xgb_safety': xgb_models,
}
test_proba, _ = predict_stacked(
    models_dict,
    test_df,
    feature_cols,
    meta_model,
    meta_scaler
)

# Apply conservative shifting
test_final_preds = conservative_shift(test_proba, entropy_threshold=best_threshold)

# Create submission
submission = pd.DataFrame({
    'patient_id': test_df['patient_id'],
    'triage_acuity': test_final_preds.astype(int)
})

submission_path = os.path.join(OUTPUT_DIR, 'submission.csv')
submission.to_csv(submission_path, index=False)

print(f"\n  Submission saved to: {submission_path}")
print(f"  Submission shape: {submission.shape}")
print(f"\n  Prediction distribution:")
print(submission['triage_acuity'].value_counts().sort_index().to_string())

# Verify format matches sample
print(f"\n  Sample submission format: {list(sample_sub.columns)}")
print(f"  Our submission format:    {list(submission.columns)}")
assert list(submission.columns) == list(sample_sub.columns), "Submission format mismatch!"
assert len(submission) == len(sample_sub), "Submission length mismatch!"
print("  [OK] Submission format verified!")


# ===============================================================================
# DONE
# ===============================================================================

print("\n\n" + "=" * 70)
print("  SafeTriageNet -- PIPELINE COMPLETE")
print("=" * 70)
print(f"""
  Outputs saved to: {OUTPUT_DIR}
  
  Files:
    * submission.csv         -- Final test predictions
    * model_comparison.csv   -- Head-to-head model metrics
    * feature_importance.csv -- Full feature ranking
    * fig1-fig11             -- Publication-quality figures
  
  Final Model: SafeTriageNet (Stacked Ensemble + Conservative Shift)
    Accuracy:     {final_report['accuracy']:.4f}
    Macro F1:     {final_report['macro_f1']:.4f}
    Under-triage: {final_report['undertriage_rate']:.4f}
    CWE:          {final_report['cost_weighted_error']:.4f}
""")
