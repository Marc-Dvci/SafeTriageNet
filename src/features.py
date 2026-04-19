"""
SafeTriageNet -- Feature Engineering Module
==========================================
Clinical-grade feature engineering for emergency triage prediction.
Transforms raw ED intake data into clinically meaningful features,
with special emphasis on informative missingness and derived physiological indices.

References:
    - Emergency Severity Index (ESI) v4 Implementation Handbook, AHRQ
    - Royal College of Physicians, National Early Warning Score (NEWS2), 2017
    - Shock Index: Allgower & Burri, 1967; Rady et al., 1994
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


# --- Clinical Constants ---------------------------------------------------------

# Age-adjusted normal vital sign ranges (approximate clinical reference)
VITAL_NORMS = {
    'pediatric':     {'age_range': (0, 17),  'hr': (70, 120), 'rr': (16, 24), 'sbp': (90, 120),  'temp': (36.1, 37.8)},
    'young_adult':   {'age_range': (18, 39), 'hr': (60, 100), 'rr': (12, 20), 'sbp': (100, 130), 'temp': (36.1, 37.8)},
    'middle_aged':   {'age_range': (40, 64), 'hr': (60, 100), 'rr': (12, 20), 'sbp': (100, 140), 'temp': (36.1, 37.8)},
    'elderly':       {'age_range': (65, 120),'hr': (60, 100), 'rr': (12, 20), 'sbp': (100, 150), 'temp': (36.1, 37.5)},
}

# High-acuity chief complaint keywords (from ESI triage literature)
HIGH_ACUITY_KEYWORDS = [
    'chest pain', 'shortness of breath', 'difficulty breathing', 'dyspnea',
    'altered mental', 'unresponsive', 'unconscious', 'seizure', 'stroke',
    'cardiac arrest', 'anaphylaxis', 'choking', 'apnea', 'cyanosis',
    'hematemesis', 'massive bleeding', 'hemorrhage', 'trauma', 'intubat',
    'overdose', 'poisoning', 'syncope', 'collapse', 'face droop',
    'slurred speech', 'focal weakness', 'thunderclap headache',
    'suicidal', 'self-harm', 'unstable', 'pulseless',
    'respiratory distress', 'respiratory failure', 'status epilepticus',
    'acute abdomen', 'ruptured', 'dissection', 'testicular torsion',
    'ectopic pregnancy', 'meningitis', 'sepsis', 'wound', 'laceration',
]

MODERATE_ACUITY_KEYWORDS = [
    'abdominal pain', 'back pain', 'fever', 'vomiting', 'diarrhea',
    'headache', 'dizziness', 'fall', 'fracture', 'burn',
    'allergic reaction', 'asthma', 'wheezing', 'palpitation',
    'nosebleed', 'epistaxis', 'cellulitis', 'abscess', 'urinary',
    'dysuria', 'hematuria', 'flank pain', 'kidney stone',
    'swelling', 'dehydration', 'diabetic', 'hyperglycemia',
]

LOW_ACUITY_KEYWORDS = [
    'rash', 'itch', 'insect bite', 'sprain', 'strain', 'contusion',
    'sore throat', 'cold symptoms', 'cough', 'earache', 'toothache',
    'prescription refill', 'medication refill', 'follow-up',
    'suture removal', 'wound check', 'discharge instructions',
    'contraception', 'general health', 'information', 'advice',
]


# --- Missingness Features -------------------------------------------------------

VITAL_COLUMNS = [
    'systolic_bp', 'diastolic_bp', 'mean_arterial_pressure',
    'pulse_pressure', 'heart_rate', 'respiratory_rate',
    'temperature_c', 'spo2', 'gcs_total', 'pain_score',
    'weight_kg', 'height_cm', 'bmi', 'shock_index', 'news2_score'
]

# Core vitals that are most informative when missing
CORE_VITALS = [
    'systolic_bp', 'diastolic_bp', 'heart_rate',
    'respiratory_rate', 'temperature_c', 'spo2'
]

# Categorical columns expected in the raw intake feed
CATEGORY_COLUMNS = [
    'arrival_mode', 'arrival_day', 'arrival_season', 'shift',
    'age_group', 'sex', 'language', 'insurance_type',
    'transport_origin', 'pain_location', 'mental_status_triage',
    'chief_complaint_system'
]


def create_missingness_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create informative missingness features.
    
    The absence of vital sign documentation in the ED is clinically meaningful:
    - Low-acuity patients often have fewer vitals recorded
    - High-acuity patients in extremis may also have missing vitals (staff focused on resuscitation)
    - The *pattern* of which vitals are missing encodes triage nurse clinical judgment
    
    Returns DataFrame with new missingness feature columns appended.
    """
    df = df.copy()
    
    # 1. Binary missingness indicators for each vital
    for col in VITAL_COLUMNS:
        if col in df.columns:
            # Handle both NaN and -1 encoding
            if col == 'pain_score':
                df[f'{col}_missing'] = (df[col].isna() | (df[col] == -1)).astype(int)
            else:
                df[f'{col}_missing'] = df[col].isna().astype(int)
    
    # 2. Documentation Completeness Score (fraction of core vitals recorded)
    missing_cols = [f'{c}_missing' for c in CORE_VITALS if f'{c}_missing' in df.columns]
    if missing_cols:
        df['doc_completeness'] = 1.0 - df[missing_cols].mean(axis=1)
    
    # 3. Total missing vitals count
    all_missing_cols = [c for c in df.columns if c.endswith('_missing')]
    if all_missing_cols:
        df['n_missing_vitals'] = df[all_missing_cols].sum(axis=1)
    
    # 4. Missingness pattern hash -- each unique pattern of missing/present is a categorical feature
    if missing_cols:
        df['missingness_pattern'] = df[missing_cols].astype(str).agg(''.join, axis=1)
    
    return df


# --- Derived Clinical Features --------------------------------------------------

def create_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive clinically meaningful physiological indices from raw vitals.
    
    These composite scores are standard clinical decision tools that compress
    multiple vital signs into single risk indicators. They are well-validated
    in the emergency medicine literature.
    """
    df = df.copy()
    
    # --- Shock Index (SI) ---
    # SI = HR / SBP. Normal: 0.5-0.7. Elevated (>0.9) strongly predicts hemodynamic instability.
    # Already provided in dataset but we recompute for robustness
    if 'heart_rate' in df.columns and 'systolic_bp' in df.columns:
        mask = (df['systolic_bp'].notna()) & (df['systolic_bp'] > 0) & (df['heart_rate'].notna())
        df['shock_index_calc'] = np.nan
        df.loc[mask, 'shock_index_calc'] = df.loc[mask, 'heart_rate'] / df.loc[mask, 'systolic_bp']
        df['shock_index_elevated'] = (df['shock_index_calc'] > 0.9).astype(float)
        df['shock_index_critical'] = (df['shock_index_calc'] > 1.3).astype(float)
    
    # --- Mean Arterial Pressure (MAP) ---
    # MAP = DBP + (SBP - DBP) / 3. Critical threshold: MAP < 65 mmHg
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        mask = df['systolic_bp'].notna() & df['diastolic_bp'].notna()
        df['map_calc'] = np.nan
        df.loc[mask, 'map_calc'] = (
            df.loc[mask, 'diastolic_bp'] + 
            (df.loc[mask, 'systolic_bp'] - df.loc[mask, 'diastolic_bp']) / 3
        )
        df['map_critical'] = (df['map_calc'] < 65).astype(float)
    
    # --- Pulse Pressure ---
    # PP = SBP - DBP. Narrow PP (<25) -> cardiogenic shock. Wide PP (>60) -> sepsis, aortic regurgitation
    if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns:
        mask = df['systolic_bp'].notna() & df['diastolic_bp'].notna()
        df['pulse_pressure_calc'] = np.nan
        df.loc[mask, 'pulse_pressure_calc'] = df.loc[mask, 'systolic_bp'] - df.loc[mask, 'diastolic_bp']
        df['pp_narrow'] = (df['pulse_pressure_calc'] < 25).astype(float)
        df['pp_wide'] = (df['pulse_pressure_calc'] > 60).astype(float)
    
    # --- SpO2 severity tiers ---
    if 'spo2' in df.columns:
        df['spo2_critical'] = (df['spo2'] < 90).astype(float)
        df['spo2_concerning'] = ((df['spo2'] >= 90) & (df['spo2'] < 94)).astype(float)
        df['spo2_normal'] = (df['spo2'] >= 94).astype(float)
    
    # --- GCS severity tiers ---
    if 'gcs_total' in df.columns:
        df['gcs_severe'] = (df['gcs_total'] <= 8).astype(float)
        df['gcs_moderate'] = ((df['gcs_total'] > 8) & (df['gcs_total'] <= 12)).astype(float)
        df['gcs_mild'] = (df['gcs_total'] == 15).astype(float)  # Normal
    
    # --- Temperature severity ---
    if 'temperature_c' in df.columns:
        df['hypothermia'] = (df['temperature_c'] < 35.5).astype(float)
        df['fever'] = (df['temperature_c'] >= 38.0).astype(float)
        df['high_fever'] = (df['temperature_c'] >= 39.0).astype(float)
        df['hyperthermia'] = (df['temperature_c'] >= 40.0).astype(float)
    
    # --- Heart rate severity ---
    if 'heart_rate' in df.columns:
        df['bradycardia'] = (df['heart_rate'] < 50).astype(float)
        df['tachycardia'] = (df['heart_rate'] > 100).astype(float)
        df['severe_tachycardia'] = (df['heart_rate'] > 130).astype(float)
    
    # --- Respiratory rate severity ---
    if 'respiratory_rate' in df.columns:
        df['tachypnea'] = (df['respiratory_rate'] > 22).astype(float)
        df['severe_tachypnea'] = (df['respiratory_rate'] > 30).astype(float)
        df['bradypnea'] = (df['respiratory_rate'] < 10).astype(float)
    
    # --- Blood pressure severity ---
    if 'systolic_bp' in df.columns:
        df['hypotension'] = (df['systolic_bp'] < 90).astype(float)
        df['severe_hypotension'] = (df['systolic_bp'] < 70).astype(float)
        df['hypertensive_urgency'] = (df['systolic_bp'] > 180).astype(float)
    
    # --- Pain severity tiers ---
    if 'pain_score' in df.columns:
        df['pain_none'] = (df['pain_score'] == 0).astype(float)
        df['pain_mild'] = ((df['pain_score'] >= 1) & (df['pain_score'] <= 3)).astype(float)
        df['pain_moderate'] = ((df['pain_score'] >= 4) & (df['pain_score'] <= 6)).astype(float)
        df['pain_severe'] = ((df['pain_score'] >= 7) & (df['pain_score'] <= 10)).astype(float)
    
    # --- Age-adjusted vital sign abnormality flags ---
    if 'age' in df.columns and 'heart_rate' in df.columns:
        df['hr_age_abnormal'] = 0.0
        for group, norms in VITAL_NORMS.items():
            lo, hi = norms['age_range']
            hr_lo, hr_hi = norms['hr']
            mask = (df['age'] >= lo) & (df['age'] <= hi) & df['heart_rate'].notna()
            df.loc[mask & ((df['heart_rate'] < hr_lo) | (df['heart_rate'] > hr_hi)), 'hr_age_abnormal'] = 1.0
    
    # --- Composite critical count ---
    # How many critical flags does this patient have? Strong ordinal signal.
    critical_flags = [c for c in df.columns if c in [
        'shock_index_critical', 'map_critical', 'spo2_critical',
        'gcs_severe', 'hypothermia', 'hyperthermia', 'severe_tachycardia',
        'severe_tachypnea', 'severe_hypotension', 'bradypnea'
    ]]
    if critical_flags:
        df['n_critical_flags'] = df[critical_flags].sum(axis=1)
    
    return df


# --- Temporal Features -----------------------------------------------------------

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode temporal patterns from arrival time and day.
    
    ED volumes, case mix, and staffing patterns vary systematically by time of day
    and day of week. Night and weekend shifts are associated with higher acuity
    casemix and potentially higher triage error rates.
    """
    df = df.copy()
    
    # Cyclical encoding of hour (captures the circular nature of time)
    if 'arrival_hour' in df.columns:
        df['hour_sin'] = np.sin(2 * np.pi * df['arrival_hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['arrival_hour'] / 24)
        
        # Clinical shift categories
        df['is_night_shift'] = ((df['arrival_hour'] >= 23) | (df['arrival_hour'] < 7)).astype(int)
        df['is_evening_shift'] = ((df['arrival_hour'] >= 15) & (df['arrival_hour'] < 23)).astype(int)
    
    # Day of week encoding
    if 'arrival_day' in df.columns:
        day_map = {
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }
        df['day_numeric'] = df['arrival_day'].map(day_map)
        df['day_sin'] = np.sin(2 * np.pi * df['day_numeric'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_numeric'] / 7)
        df['is_weekend'] = (df['arrival_day'].isin(['Saturday', 'Sunday'])).astype(int)
    
    # Arrival month cyclical
    if 'arrival_month' in df.columns:
        df['month_sin'] = np.sin(2 * np.pi * df['arrival_month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['arrival_month'] / 12)
    
    return df


# --- NLP Features ----------------------------------------------------------------

def create_nlp_features(df: pd.DataFrame, complaint_col: str = 'chief_complaint_raw') -> pd.DataFrame:
    """
    Extract clinically meaningful features from free-text chief complaint narratives.
    
    Uses keyword-based clinical classification rather than heavy NLP models,
    making the approach interpretable and reproducible without GPU requirements.
    """
    df = df.copy()
    
    if complaint_col not in df.columns:
        return df
    
    text = df[complaint_col].fillna('').str.lower()
    
    # Complaint length (very short complaints from unconscious/critical patients)
    df['complaint_length'] = text.str.len()
    df['complaint_word_count'] = text.str.split().str.len().fillna(0)
    
    # High-acuity keyword match count  
    df['n_high_acuity_keywords'] = 0
    for keyword in HIGH_ACUITY_KEYWORDS:
        df['n_high_acuity_keywords'] += text.str.contains(keyword, na=False).astype(int)
    
    # Moderate-acuity keyword match count
    df['n_moderate_acuity_keywords'] = 0
    for keyword in MODERATE_ACUITY_KEYWORDS:
        df['n_moderate_acuity_keywords'] += text.str.contains(keyword, na=False).astype(int)
    
    # Low-acuity keyword match count
    df['n_low_acuity_keywords'] = 0
    for keyword in LOW_ACUITY_KEYWORDS:
        df['n_low_acuity_keywords'] += text.str.contains(keyword, na=False).astype(int)
    
    # Acuity keyword balance
    df['keyword_acuity_signal'] = df['n_high_acuity_keywords'] - df['n_low_acuity_keywords']
    
    # Specific high-risk flags
    df['cc_chest_pain'] = text.str.contains('chest pain|chest tightness|angina', na=False).astype(int)
    df['cc_sob'] = text.str.contains('shortness of breath|difficulty breathing|dyspnea|breathless', na=False).astype(int)
    df['cc_ams'] = text.str.contains('altered mental|confusion|confused|disoriented|unresponsive|unconscious', na=False).astype(int)
    df['cc_trauma'] = text.str.contains('trauma|mva|motor vehicle|assault|fall from|stabbing|gunshot|laceration|wound', na=False).astype(int)
    df['cc_neuro'] = text.str.contains('seizure|stroke|weakness|numbness|paralysis|face droop|slurred|aphasia|thunderclap', na=False).astype(int)
    df['cc_cardiac'] = text.str.contains('cardiac|heart attack|palpitation|arrhythmia|chest pain', na=False).astype(int)
    df['cc_gi_bleed'] = text.str.contains('blood in stool|hematemesis|melena|rectal bleeding|vomiting blood', na=False).astype(int)
    df['cc_mental_health'] = text.str.contains('suicid|self-harm|overdose|psychiatric|psychosis', na=False).astype(int)
    df['cc_respiratory'] = text.str.contains('asthma|wheezing|copd|respiratory|croup|stridor', na=False).astype(int)
    df['cc_sepsis'] = text.str.contains('sepsis|septic|infection.*fever|fever.*infection', na=False).astype(int)
    df['cc_abdominal'] = text.str.contains('abdominal pain|stomach pain|belly pain|acute abdomen', na=False).astype(int)
    
    # Urgency modifiers
    df['cc_acute'] = text.str.contains('acute|sudden|abrupt|worst|severe|excruciating|unbearable|worsening', na=False).astype(int)
    df['cc_chronic'] = text.str.contains('chronic|ongoing|recurrent|intermittent|long-standing|months|years', na=False).astype(int)
    
    return df


# --- Comorbidity Features --------------------------------------------------------

def create_comorbidity_features(df: pd.DataFrame, history_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Aggregate patient history flags into clinically meaningful risk composites.
    """
    df = df.copy()
    
    if history_df is not None:
        # Merge patient history
        hx_cols = [c for c in history_df.columns if c.startswith('hx_')]
        df = df.merge(history_df[['patient_id'] + hx_cols], on='patient_id', how='left')
    
    hx_cols = [c for c in df.columns if c.startswith('hx_')]
    
    if hx_cols:
        # Total comorbidity burden
        df['total_hx_flags'] = df[hx_cols].sum(axis=1)
        
        # Cardiovascular risk cluster
        cv_cols = [c for c in hx_cols if c in [
            'hx_hypertension', 'hx_heart_failure', 'hx_atrial_fibrillation',
            'hx_coronary_artery_disease', 'hx_peripheral_vascular_disease', 'hx_stroke_prior'
        ]]
        if cv_cols:
            df['hx_cardiovascular_burden'] = df[cv_cols].sum(axis=1)
        
        # Metabolic risk cluster
        met_cols = [c for c in hx_cols if c in [
            'hx_diabetes_type1', 'hx_diabetes_type2', 'hx_obesity',
            'hx_hypothyroidism', 'hx_hyperthyroidism'
        ]]
        if met_cols:
            df['hx_metabolic_burden'] = df[met_cols].sum(axis=1)
        
        # Immunocompromised risk
        immune_cols = [c for c in hx_cols if c in [
            'hx_immunosuppressed', 'hx_hiv', 'hx_malignancy'
        ]]
        if immune_cols:
            df['hx_immunocompromised'] = (df[immune_cols].sum(axis=1) > 0).astype(int)
        
        # Respiratory risk
        resp_cols = [c for c in hx_cols if c in ['hx_asthma', 'hx_copd']]
        if resp_cols:
            df['hx_respiratory_risk'] = df[resp_cols].sum(axis=1)
        
        # Mental health risk
        mh_cols = [c for c in hx_cols if c in [
            'hx_depression', 'hx_anxiety', 'hx_substance_use_disorder'
        ]]
        if mh_cols:
            df['hx_mental_health_burden'] = df[mh_cols].sum(axis=1)
        
        # Bleeding risk
        if 'hx_coagulopathy' in df.columns:
            df['hx_bleeding_risk'] = df['hx_coagulopathy']
        
        # High complexity flag (>= 5 comorbidities)
        df['hx_high_complexity'] = (df['total_hx_flags'] >= 5).astype(int)
    
    return df


# --- Master Pipeline ------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    complaints_df: Optional[pd.DataFrame] = None,
    history_df: Optional[pd.DataFrame] = None,
    is_train: bool = True
) -> pd.DataFrame:
    """
    Master feature engineering pipeline. Applies all feature transforms in order.
    
    Parameters:
        df: Main patient intake DataFrame (train.csv or test.csv)
        complaints_df: Chief complaints DataFrame
        history_df: Patient history DataFrame  
        is_train: Whether this is training data (controls target handling)
    
    Returns:
        DataFrame with all engineered features
    """
    print(f"[Features] Starting feature engineering on {len(df)} records...")
    
    # Merge chief complaints if provided
    if complaints_df is not None:
        df = df.merge(complaints_df[['patient_id', 'chief_complaint_raw']], 
                       on='patient_id', how='left')
        print(f"  -> Merged chief complaints")
    
    # Handle pain_score -1 encoding -> NaN
    if 'pain_score' in df.columns:
        df['pain_score_reported'] = (df['pain_score'] != -1).astype(int)
        df.loc[df['pain_score'] == -1, 'pain_score'] = np.nan
    
    # Apply feature pipelines
    df = create_missingness_features(df)
    print(f"  -> Missingness features created")
    
    df = create_clinical_features(df)
    print(f"  -> Clinical features created")
    
    df = create_temporal_features(df)
    print(f"  -> Temporal features created")
    
    df = create_nlp_features(df)
    print(f"  -> NLP features created")
    
    df = create_comorbidity_features(df, history_df)
    print(f"  -> Comorbidity features created")
    
    total_features = len([c for c in df.columns if c not in ['patient_id', 'triage_acuity']])
    print(f"[Features] Complete. Total features: {total_features}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Return the list of feature columns suitable for modeling.
    Excludes IDs, targets, raw text, and intermediate columns.
    """
    exclude = {
        'patient_id', 'triage_acuity', 'chief_complaint_raw',
        'missingness_pattern', 'chief_complaint_system',
        'disposition', 'ed_los_hours',  # leakage: these are post-triage outcomes
        'site_id', 'triage_nurse_id',   # bias: institutional/nurse-specific patterns
    }
    
    # Get all numeric and boolean columns
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in ['float64', 'float32', 'int64', 'int32', 'uint8', 'bool']:
            feature_cols.append(col)
    
    return feature_cols


def build_category_maps(df: pd.DataFrame) -> Dict[str, List]:
    """
    Fit categorical vocabularies on a reference DataFrame.

    These mappings should be learned on the training split and then reused for
    validation/test data so encoded category IDs remain stable.
    """
    category_maps: Dict[str, List] = {}

    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            values = pd.Series(df[col]).dropna().unique().tolist()
            category_maps[col] = values

    return category_maps


def encode_categoricals(
    df: pd.DataFrame,
    category_maps: Optional[Dict[str, List]] = None
) -> pd.DataFrame:
    """
    Encode categorical columns for tree-based models.

    When ``category_maps`` is provided, categories are encoded against that
    fixed vocabulary. Unseen values are mapped to ``-1``.
    """
    df = df.copy()

    if category_maps is None:
        category_maps = build_category_maps(df)

    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            categories = category_maps.get(col)
            if categories is None:
                categories = pd.Series(df[col]).dropna().unique().tolist()
            df[col] = pd.Categorical(df[col], categories=categories).codes.astype(np.int32)

    return df
