"""
SafeTriageNet -- Clinical Safety Module
=======================================
Asymmetric clinical cost matrix and safety-aware evaluation metrics
for emergency triage prediction.

The core insight: in emergency triage, under-triage (classifying a critically ill
patient as non-urgent) is far more dangerous than over-triage (classifying a
stable patient as urgent). Standard accuracy metrics treat both errors equally.
This module implements clinically-grounded asymmetric evaluation.

References:
    - Mistry et al., "Accuracy of Emergency Severity Index Triage," Annals of
      Emergency Medicine, 2018
    - Hinson et al., "Triage Performance in Emergency Medicine," Annals of
      Emergency Medicine, 2019
    - American College of Surgeons Committee on Trauma, "Resources for Optimal
      Care of the Injured Patient," 2014 (under-triage rate < 5% criterion)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report
)


# --- Asymmetric Clinical Cost Matrix --------------------------------------------

# ESI levels: 1 (Resuscitation) -> 5 (Non-urgent)
# Matrix[actual][predicted] = clinical cost of that misclassification
# Design principles:
#   1. Under-triage (actual < predicted, i.e., more urgent than predicted) costs MORE
#   2. Cost scales super-linearly with distance from diagonal
#   3. Under-triage of ESI-1 patients is catastrophic (life-threatening delay)
#   4. Over-triage has non-zero cost (wastes scarce resuscitation resources)

CLINICAL_COST_MATRIX = np.array([
    #  Pred:  1     2     3     4     5
    [  0.0,  1.0,  4.0, 10.0, 20.0],   # Actual: 1 (Resuscitation)
    [  0.5,  0.0,  2.0,  6.0, 12.0],   # Actual: 2 (Emergent)
    [  0.3,  0.5,  0.0,  3.0,  8.0],   # Actual: 3 (Urgent)
    [  0.2,  0.3,  0.5,  0.0,  3.0],   # Actual: 4 (Less Urgent)
    [  0.1,  0.2,  0.3,  0.5,  0.0],   # Actual: 5 (Non-urgent)
], dtype=np.float64)


def get_cost_matrix() -> np.ndarray:
    """Return the asymmetric clinical cost matrix (5x5, 0-indexed)."""
    return CLINICAL_COST_MATRIX.copy()


def cost_weighted_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute the mean cost-weighted error using the clinical cost matrix.
    
    Parameters:
        y_true: Array of true ESI labels (1-5)
        y_pred: Array of predicted ESI labels (1-5)
    
    Returns:
        Mean clinical cost across all predictions. Lower is better.
    """
    costs = np.array([
        CLINICAL_COST_MATRIX[int(t) - 1, int(p) - 1]
        for t, p in zip(y_true, y_pred)
    ])
    return costs.mean()


def undertriage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute under-triage rate: fraction of high-acuity patients (ESI 1-2)
    incorrectly predicted as lower acuity (ESI 3-5).
    
    This is the primary patient safety metric in triage evaluation.
    The American College of Surgeons recommends an under-triage rate < 5%.
    """
    high_acuity_mask = np.isin(y_true, [1, 2])
    if high_acuity_mask.sum() == 0:
        return 0.0
    
    high_acuity_true = y_true[high_acuity_mask]
    high_acuity_pred = y_pred[high_acuity_mask]
    
    undertriaged = high_acuity_pred > 2  # Predicted 3, 4, or 5
    return undertriaged.mean()


def overtriage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute over-triage rate: fraction of low-acuity patients (ESI 4-5)
    incorrectly predicted as higher acuity (ESI 1-3).
    
    Over-triage wastes resources but doesn't directly endanger patients.
    Some over-triage is acceptable and even desirable from a safety standpoint.
    """
    low_acuity_mask = np.isin(y_true, [4, 5])
    if low_acuity_mask.sum() == 0:
        return 0.0
    
    low_acuity_true = y_true[low_acuity_mask]
    low_acuity_pred = y_pred[low_acuity_mask]
    
    overtriaged = low_acuity_pred < 4  # Predicted 1, 2, or 3
    return overtriaged.mean()


def severe_undertriage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Severe under-triage: ESI 1-2 patients predicted as ESI 4-5.
    This is the most dangerous error class -- off by >=2 levels.
    """
    high_acuity_mask = np.isin(y_true, [1, 2])
    if high_acuity_mask.sum() == 0:
        return 0.0
    
    high_acuity_pred = y_pred[high_acuity_mask]
    severely_undertriaged = np.isin(high_acuity_pred, [4, 5])
    return severely_undertriaged.mean()


def safety_adjusted_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Weighted Cohen's Kappa using clinical cost-derived weights.
    
    Standard Cohen's Kappa uses either linear or quadratic weights.
    We use the clinical cost matrix as custom weights, so disagreements
    that are clinically dangerous contribute more to the score.
    """
    # Create a weight matrix from costs (normalized to 0-1 range)
    max_cost = CLINICAL_COST_MATRIX.max()
    weights = CLINICAL_COST_MATRIX / max_cost
    
    # Use sklearn's kappa with custom sample weights isn't directly supported,
    # so we compute via the confusion matrix
    n_classes = 5
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    
    n = cm.sum()
    if n == 0:
        return 0.0
    
    # Expected frequency under chance
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    expected = np.outer(row_sums, col_sums) / n
    
    # Weighted observed and expected disagreement
    w_observed = (weights * cm).sum() / n
    w_expected = (weights * expected).sum() / n
    
    if w_expected == 0:
        return 1.0
    
    return 1.0 - (w_observed / w_expected)


def adjacent_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Fraction of predictions within +/-1 of the true ESI level.
    Clinically, being off by 1 level is often tolerable.
    """
    return (np.abs(y_true - y_pred) <= 1).mean()


# --- Comprehensive Safety Report ------------------------------------------------

def full_safety_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
    model_name: str = "Model"
) -> Dict:
    """
    Generate a comprehensive evaluation report with both standard ML metrics
    and clinical safety metrics.
    
    Returns a dictionary of all metrics.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    
    report = {
        'model_name': model_name,
        
        # Standard ML metrics
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'quadratic_kappa': cohen_kappa_score(y_true, y_pred, weights='quadratic'),
        'adjacent_accuracy': adjacent_accuracy(y_true, y_pred),
        
        # Clinical safety metrics
        'undertriage_rate': undertriage_rate(y_true, y_pred),
        'severe_undertriage_rate': severe_undertriage_rate(y_true, y_pred),
        'overtriage_rate': overtriage_rate(y_true, y_pred),
        'cost_weighted_error': cost_weighted_error(y_true, y_pred),
        'safety_adjusted_kappa': safety_adjusted_kappa(y_true, y_pred),
    }
    
    # Per-class metrics
    per_class = classification_report(y_true, y_pred, labels=[1, 2, 3, 4, 5],
                                       output_dict=True, zero_division=0)
    for esi in [1, 2, 3, 4, 5]:
        key = str(esi)
        if key in per_class:
            report[f'esi{esi}_precision'] = per_class[key]['precision']
            report[f'esi{esi}_recall'] = per_class[key]['recall']
            report[f'esi{esi}_f1'] = per_class[key]['f1-score']
    
    return report


def print_safety_report(report: Dict):
    """Pretty-print a safety report."""
    print(f"\n{'='*70}")
    print(f"  EVALUATION REPORT: {report['model_name']}")
    print(f"{'='*70}")
    
    print(f"\n  -- Standard ML Metrics --")
    print(f"  Accuracy:             {report['accuracy']:.4f}")
    print(f"  Macro F1:             {report['macro_f1']:.4f}")
    print(f"  Weighted F1:          {report['weighted_f1']:.4f}")
    print(f"  Cohen's Kappa:        {report['cohen_kappa']:.4f}")
    print(f"  Quadratic Kappa:      {report['quadratic_kappa']:.4f}")
    print(f"  Adjacent Accuracy:    {report['adjacent_accuracy']:.4f}")
    
    print(f"\n  -- Clinical Safety Metrics --")
    print(f"  Under-triage Rate:    {report['undertriage_rate']:.4f}  "
          f"({'[!] HIGH' if report['undertriage_rate'] > 0.05 else '[OK] SAFE'})")
    print(f"  Severe Under-triage:  {report['severe_undertriage_rate']:.4f}  "
          f"({'[!!] CRITICAL' if report['severe_undertriage_rate'] > 0.02 else '[OK] SAFE'})")
    print(f"  Over-triage Rate:     {report['overtriage_rate']:.4f}")
    print(f"  Cost-Weighted Error:  {report['cost_weighted_error']:.4f}")
    print(f"  Safety-Adj. Kappa:    {report['safety_adjusted_kappa']:.4f}")
    
    print(f"\n  -- Per-Class Performance --")
    print(f"  {'ESI':<6} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*42}")
    for esi in [1, 2, 3, 4, 5]:
        p = report.get(f'esi{esi}_precision', 0)
        r = report.get(f'esi{esi}_recall', 0)
        f = report.get(f'esi{esi}_f1', 0)
        label = ['Resuscitation', 'Emergent', 'Urgent', 'Less Urgent', 'Non-urgent'][esi-1]
        print(f"  ESI-{esi} ({label:<14}) {p:<12.4f} {r:<12.4f} {f:<12.4f}")
    
    print(f"\n{'='*70}\n")


# --- Custom LightGBM Loss Function ----------------------------------------------

def asymmetric_multiclass_objective(y_pred_raw: np.ndarray, dataset) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom LightGBM objective function with asymmetric clinical cost penalization.
    
    This function penalizes under-triage (predicting lower acuity than actual)
    more heavily than over-triage, using the clinical cost matrix as weights.
    
    Note: This is applied to the softmax-transformed outputs of a multiclass model.
    For LightGBM, y_pred_raw is flattened: (n_samples * n_classes,) with class-major ordering.
    """
    labels = dataset.get_label().astype(int)
    n_samples = len(labels)
    n_classes = 5
    
    # Reshape predictions to (n_samples, n_classes)
    y_pred_raw = y_pred_raw.reshape(n_classes, n_samples).T
    
    # Softmax
    exp_pred = np.exp(y_pred_raw - y_pred_raw.max(axis=1, keepdims=True))
    softmax = exp_pred / exp_pred.sum(axis=1, keepdims=True)
    
    # One-hot encode labels (0-indexed)
    labels_0idx = labels - 1  # Convert 1-5 to 0-4
    one_hot = np.zeros((n_samples, n_classes))
    one_hot[np.arange(n_samples), labels_0idx] = 1
    
    # Standard cross-entropy gradient: (softmax - one_hot)
    # We weight each class's gradient contribution by the clinical cost
    grad = softmax - one_hot
    
    # Apply asymmetric cost weighting
    for i in range(n_samples):
        true_class = labels_0idx[i]
        for c in range(n_classes):
            if c != true_class:
                cost = CLINICAL_COST_MATRIX[true_class, c]
                grad[i, c] *= (1.0 + cost)  # Scale gradient by cost
    
    # Hessian (second derivative of cross-entropy): softmax * (1 - softmax)
    hessian = softmax * (1.0 - softmax)
    
    # Apply same cost weighting to hessian
    for i in range(n_samples):
        true_class = labels_0idx[i]
        for c in range(n_classes):
            if c != true_class:
                cost = CLINICAL_COST_MATRIX[true_class, c]
                hessian[i, c] *= (1.0 + cost)
    
    # Flatten back to (n_classes * n_samples,)
    grad = grad.T.flatten()
    hessian = hessian.T.flatten()
    
    return grad, hessian


def conservative_shift(
    y_pred_proba: np.ndarray,
    entropy_threshold: float = 0.7,
    shift_strength: int = 1
) -> np.ndarray:
    """
    Uncertainty-aware conservative prediction shifting.
    
    When the model's prediction entropy exceeds a threshold (indicating uncertainty),
    the prediction is shifted toward higher acuity (lower ESI number) to err on
    the side of patient safety.
    
    Parameters:
        y_pred_proba: (n_samples, 5) array of class probabilities for ESI 1-5
        entropy_threshold: Normalized entropy threshold (0-1) above which to shift
        shift_strength: Number of ESI levels to shift toward higher acuity
    
    Returns:
        Array of adjusted predictions (1-5)
    """
    # Compute prediction entropy (normalized to 0-1)
    eps = 1e-10
    entropy = -np.sum(y_pred_proba * np.log(y_pred_proba + eps), axis=1)
    max_entropy = np.log(y_pred_proba.shape[1])
    normalized_entropy = entropy / max_entropy
    
    # Base predictions (argmax + 1 for ESI 1-5)
    predictions = y_pred_proba.argmax(axis=1) + 1
    
    # Shift uncertain predictions toward higher acuity
    uncertain_mask = normalized_entropy > entropy_threshold
    predictions[uncertain_mask] = np.maximum(
        1, predictions[uncertain_mask] - shift_strength
    )
    
    return predictions
