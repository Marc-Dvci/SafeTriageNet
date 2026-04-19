"""
SafeTriageNet -- Modeling Module
================================
Multi-model training, stacking, and safety-aware prediction for emergency triage.
Implements baseline and safety-aware models with proper cross-validation.

Architecture:
    Tier 1: Independent sub-models (LightGBM, XGBoost)
    Tier 2: Stacking meta-learner with regularized multinomial logistic regression
    Tier 3: Safety-aware post-processing with conservative shifting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# --- LightGBM Configuration ----------------------------------------------------

LGBM_BASE_PARAMS = {
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
}

# Class weights to address class imbalance (ESI-1 is rare, ESI-3 is dominant)
# These give MORE weight to rare, high-acuity classes
ACUITY_SAMPLE_WEIGHTS = {
    1: 5.0,   # Resuscitation -- rare, must not miss
    2: 3.0,   # Emergent
    3: 1.0,   # Urgent -- most common
    4: 1.5,   # Less urgent
    5: 2.0,   # Non-urgent -- also needs correct identification
}

XGB_BASE_PARAMS = {
    'objective': 'multi:softprob',
    'num_class': 5,
    'eval_metric': 'mlogloss',
    'max_depth': 7,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'min_child_weight': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'tree_method': 'hist',
    'seed': 42,
    'verbosity': 0,
}


# --- Training Utilities --------------------------------------------------------

def create_sample_weights(y: np.ndarray) -> np.ndarray:
    """Create sample weights from class-aware weighting scheme."""
    return np.array([ACUITY_SAMPLE_WEIGHTS.get(label, 1.0) for label in y])


def _build_meta_model() -> LogisticRegression:
    """Create a fresh stacking meta-learner instance."""
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
    )


def train_lgbm_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    n_folds: int = 5,
    n_boost_rounds: int = 1500,
    early_stopping_rounds: int = 50,
    params: Optional[Dict] = None,
    use_sample_weights: bool = True,
    model_name: str = "LightGBM"
) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Train LightGBM with stratified K-fold cross-validation.
    
    Returns:
        models: List of trained LightGBM Booster models
        oof_preds: Out-of-fold probability predictions (n_samples, 5)
        oof_labels: Out-of-fold predicted labels (n_samples,)
    """
    if params is None:
        params = LGBM_BASE_PARAMS.copy()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(X), 5))
    models = []
    
    print(f"\n{'-'*60}")
    print(f"  Training {model_name} ({n_folds}-fold CV)")
    print(f"{'-'*60}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx][feature_cols]
        y_train = y[train_idx]
        X_val = X.iloc[val_idx][feature_cols]
        y_val = y[val_idx]
        
        # Convert labels from 1-5 to 0-4 for LightGBM
        y_train_0idx = y_train - 1
        y_val_0idx = y_val - 1
        
        # Create sample weights
        w_train = create_sample_weights(y_train) if use_sample_weights else None
        
        dtrain = lgb.Dataset(X_train, label=y_train_0idx, weight=w_train)
        dval = lgb.Dataset(X_val, label=y_val_0idx, reference=dtrain)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=0),
        ]
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=n_boost_rounds,
            valid_sets=[dval],
            callbacks=callbacks,
        )
        
        val_preds = model.predict(X_val)
        oof_preds[val_idx] = val_preds
        models.append(model)
        
        val_labels = val_preds.argmax(axis=1) + 1
        fold_acc = (val_labels == y_val).mean()
        print(f"  Fold {fold_idx + 1}/{n_folds}  |  "
              f"Best iteration: {model.best_iteration}  |  "
              f"Val Accuracy: {fold_acc:.4f}")
    
    oof_labels = oof_preds.argmax(axis=1) + 1
    overall_acc = (oof_labels == y).mean()
    print(f"\n  -> Overall OOF Accuracy: {overall_acc:.4f}")
    
    return models, oof_preds, oof_labels


def train_xgb_cv(
    X: pd.DataFrame,
    y: np.ndarray,
    feature_cols: List[str],
    n_folds: int = 5,
    n_boost_rounds: int = 1500,
    early_stopping_rounds: int = 50,
    params: Optional[Dict] = None,
    use_sample_weights: bool = True,
    model_name: str = "XGBoost"
) -> Tuple[List, np.ndarray, np.ndarray]:
    """
    Train XGBoost with stratified K-fold cross-validation.
    """
    if params is None:
        params = XGB_BASE_PARAMS.copy()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    oof_preds = np.zeros((len(X), 5))
    models = []
    
    print(f"\n{'-'*60}")
    print(f"  Training {model_name} ({n_folds}-fold CV)")
    print(f"{'-'*60}")
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train = X.iloc[train_idx][feature_cols].values
        y_train = y[train_idx]
        X_val = X.iloc[val_idx][feature_cols].values
        y_val = y[val_idx]
        
        # Convert labels from 1-5 to 0-4
        y_train_0idx = y_train - 1
        y_val_0idx = y_val - 1
        
        w_train = create_sample_weights(y_train) if use_sample_weights else None
        
        dtrain = xgb.DMatrix(X_train, label=y_train_0idx, weight=w_train)
        dval = xgb.DMatrix(X_val, label=y_val_0idx)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_boost_rounds,
            evals=[(dval, 'val')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False,
        )
        
        val_preds = model.predict(dval)
        oof_preds[val_idx] = val_preds
        models.append(model)
        
        val_labels = val_preds.argmax(axis=1) + 1
        fold_acc = (val_labels == y_val).mean()
        print(f"  Fold {fold_idx + 1}/{n_folds}  |  "
              f"Best iteration: {model.best_iteration}  |  "
              f"Val Accuracy: {fold_acc:.4f}")
    
    oof_labels = oof_preds.argmax(axis=1) + 1
    overall_acc = (oof_labels == y).mean()
    print(f"\n  -> Overall OOF Accuracy: {overall_acc:.4f}")
    
    return models, oof_preds, oof_labels


# --- Stacking Meta-Learner -----------------------------------------------------

def train_stacking_meta(
    oof_predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    n_folds: int = 5,
) -> Tuple[LogisticRegression, StandardScaler, np.ndarray, np.ndarray]:
    """
    Train a stacking meta-learner on out-of-fold predictions from base models.
    
    The meta-learner is evaluated with its own stratified cross-validation so the
    reported ensemble metrics are based on honest out-of-fold predictions rather
    than the same meta-features used for fitting.
    
    Parameters:
        oof_predictions: Dict of model_name -> (n_samples, 5) OOF probability arrays
        y_true: True labels (1-5)
        n_folds: Number of CV folds for meta-learner training
    
    Returns:
        Fitted full-data LogisticRegression model and StandardScaler, plus
        meta-learner OOF probabilities and OOF labels for evaluation
    """
    meta_features = np.hstack(list(oof_predictions.values()))
    classes = np.sort(np.unique(y_true))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    sample_weights = create_sample_weights(y_true)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_meta_proba = np.zeros((len(y_true), len(classes)))
    oof_meta_labels = np.zeros(len(y_true), dtype=int)
    
    print(f"\n{'-'*60}")
    print(f"  Training Stacking Meta-Learner")
    print(f"  Input models: {list(oof_predictions.keys())}")
    print(f"  Meta-feature dimension: {meta_features.shape[1]}")
    print(f"{'-'*60}")

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(meta_features, y_true)):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(meta_features[train_idx])
        X_val = scaler.transform(meta_features[val_idx])

        meta_model = _build_meta_model()
        meta_model.fit(
            X_train,
            y_true[train_idx],
            sample_weight=sample_weights[train_idx]
        )

        val_proba = meta_model.predict_proba(X_val)
        val_labels = meta_model.predict(X_val)

        for local_idx, class_label in enumerate(meta_model.classes_):
            oof_meta_proba[val_idx, class_to_idx[class_label]] = val_proba[:, local_idx]
        oof_meta_labels[val_idx] = val_labels

        fold_acc = (val_labels == y_true[val_idx]).mean()
        print(f"  Fold {fold_idx + 1}/{n_folds}  |  Val Accuracy: {fold_acc:.4f}")

    meta_oof_acc = (oof_meta_labels == y_true).mean()
    print(f"\n  -> Meta-learner OOF Accuracy: {meta_oof_acc:.4f}")

    full_scaler = StandardScaler()
    meta_features_scaled = full_scaler.fit_transform(meta_features)

    full_meta_model = _build_meta_model()
    full_meta_model.fit(
        meta_features_scaled,
        y_true,
        sample_weight=sample_weights
    )

    return full_meta_model, full_scaler, oof_meta_proba, oof_meta_labels


def predict_stacked(
    models_dict: Dict[str, List],
    X_test: pd.DataFrame,
    feature_cols: List[str],
    meta_model: LogisticRegression,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions from the full stacking pipeline.
    
    Parameters:
        models_dict: Dict of model_name -> list of fold models
        X_test: Test features DataFrame  
        feature_cols: Feature column names
        meta_model: Fitted meta-learner
        scaler: Fitted StandardScaler
    
    Returns:
        Probability predictions (n_samples, 5) and label predictions (n_samples,)
    """
    test_preds_list = []
    
    for model_name, fold_models in models_dict.items():
        model_preds = np.zeros((len(X_test), 5))
        
        for model in fold_models:
            if isinstance(model, lgb.Booster):
                preds = model.predict(X_test[feature_cols])
            elif isinstance(model, xgb.Booster):
                dtest = xgb.DMatrix(X_test[feature_cols].values)
                preds = model.predict(dtest)
            else:
                raise ValueError(f"Unknown model type: {type(model)}")
            
            model_preds += preds
        
        model_preds /= len(fold_models)
        test_preds_list.append(model_preds)
    
    # Stack and transform through meta-learner
    meta_features = np.hstack(test_preds_list)
    meta_features_scaled = scaler.transform(meta_features)
    
    proba = meta_model.predict_proba(meta_features_scaled)
    labels = meta_model.predict(meta_features_scaled)
    
    return proba, labels


# --- Feature Importance --------------------------------------------------------

def get_feature_importance(
    models: List,
    feature_cols: List[str],
    importance_type: str = 'gain'
) -> pd.DataFrame:
    """
    Aggregate feature importance across CV fold models.
    """
    importance_df = pd.DataFrame({'feature': feature_cols})
    
    for i, model in enumerate(models):
        if isinstance(model, lgb.Booster):
            imp = model.feature_importance(importance_type=importance_type)
        else:
            continue
        importance_df[f'fold_{i}'] = imp
    
    fold_cols = [c for c in importance_df.columns if c.startswith('fold_')]
    importance_df['mean_importance'] = importance_df[fold_cols].mean(axis=1)
    importance_df['std_importance'] = importance_df[fold_cols].std(axis=1)
    importance_df = importance_df.sort_values('mean_importance', ascending=False)
    
    return importance_df[['feature', 'mean_importance', 'std_importance']]
