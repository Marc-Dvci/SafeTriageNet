# SafeTriageNet

**Safety-Aware Multimodal Triage with Informative Missingness and Asymmetric Clinical Cost**

> *"When Getting It Wrong Matters More Than Getting It Right"*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Overview

SafeTriageNet is a safety-aware triage decision support system built for the [Triagegeist Kaggle Competition](https://kaggle.com/competitions/triagegeist), hosted by the Laitinen-Fredriksson Foundation.

Unlike standard classification approaches that optimize only for accuracy, SafeTriageNet is designed to reduce **clinically dangerous under-triage** while tracking an asymmetric error cost that treats missed critical patients as far more serious than benign over-triage.

### Three Pillars

1. **Informative Missingness** -- Documentation completeness and several missing-vital patterns carry signal about patient acuity
2. **Multimodal Feature Set** -- Structured intake data, complaint-text heuristics, and comorbidity history are engineered into one modeling table
3. **Safety-Aware Selection** -- Asymmetric cost tracking, sample weighting for rare high-acuity classes, and conservative uncertainty shifting

---

## Results

| Model | Accuracy | Macro F1 | Under-Triage Rate | Cost-Weighted Error |
|---|---|---|---|---|
| Baseline LightGBM | 0.8866 | 0.8964 | 0.0161 | 0.2047 |
| Safety-Weighted LightGBM | 0.8861 | 0.8974 | 0.0137 | 0.2312 |
| Safety-Weighted XGBoost | 0.8841 | 0.8956 | 0.0127 | 0.2349 |
| Stacked Ensemble | 0.8859 | 0.8966 | 0.0112 | 0.2256 |
| **SafeTriageNet (Final)** | **0.8858** | **0.8965** | **0.0112** | **0.2229** |

**Key result:** SafeTriageNet reduces the under-triage rate by 31% (1.61% -> 1.12%) compared to the accuracy-optimized baseline while preserving comparable overall accuracy. Severe under-triage remained at 0.0% across all audited models.

**Evaluation note:** The stacked-model metrics above are based on out-of-fold meta-learner predictions rather than in-sample meta-training scores.

**Selection note:** The baseline LightGBM retains the lowest raw cost-weighted error, but the final submission was chosen for its safer under-triage profile at essentially unchanged accuracy.

---

## Project Structure

```
SafeTriageNet/
|-- README.md                    # This file
|-- LICENSE                      # MIT License
|-- requirements.txt             # Python dependencies
|-- notebooks/
|   |-- safetriagenet.py         # Main analysis script (local dev)
|   +-- safetriagenet.ipynb      # Self-contained Kaggle notebook (no src/ dependency)
|-- src/
|   |-- __init__.py
|   |-- features.py              # Feature engineering (missingness, clinical, NLP, temporal)
|   |-- models.py                # LightGBM/XGBoost CV training, stacking meta-learner
|   +-- safety.py                # Asymmetric cost matrix, clinical safety metrics
|-- build/
|   +-- build_kaggle_notebook.py # Inlines src/ into a self-contained .ipynb
|-- outputs/
|   |-- submission.csv           # Final test predictions
|   |-- model_comparison.csv     # Head-to-head metrics
|   |-- feature_importance.csv   # Full feature ranking
|   +-- fig1-fig11*.png          # Publication-quality figures
|-- docs/
|   |-- SUBMISSION_AUDIT.md      # Audit findings, fixes, and remaining risks
|   |-- KAGGLE_WRITEUP_V1.md     # Final Kaggle project writeup (~1965 words)
|   +-- KAGGLE_WRITEUP_V0.md     # Historical draft
+-- assets/
    +-- cover_image.png          # Competition cover image (560x280)
```

---

## Setup & Reproduction

```bash
# Clone the repository
git clone https://github.com/Marc-Dvci/SafeTriageNet.git
cd SafeTriageNet

# Install dependencies
pip install -r requirements.txt

# Place Triagegeist data in ../triagegeist/
# (train.csv, test.csv, chief_complaints.csv, patient_history.csv, sample_submission.csv)
# Or point the script at another location:
# export TRIAGEGEIST_DATA_DIR=/path/to/triagegeist

# Run the full pipeline
python -u notebooks/safetriagenet.py
```

**Requirements:** Python 3.10+, ~8GB RAM, ~3 minutes on modern CPU.

---

## Methodology

### Feature Engineering

The pipeline creates 153 engineered columns and uses 146 modeling features after excluding IDs, raw text, and obvious leakage variables (`disposition`, `ed_los_hours`, `site_id`, `triage_nurse_id`).

- **Raw physiology and intake context**: blood pressure, heart rate, respiratory rate, temperature, SpO2, GCS, pain score, age, arrival mode, prior utilization counts
- **Derived clinical features**: shock index, MAP, pulse pressure, temperature / respiratory / hemodynamic abnormality flags, age-adjusted heart-rate abnormality, critical-flag counts
- **Informative missingness**: per-vital missing flags, documentation completeness, total missing-vital count
- **Chief complaint NLP heuristics**: complaint length, keyword acuity counts, targeted high-risk complaint flags
- **Comorbidity composites**: cardiovascular, metabolic, respiratory, mental health, and immunocompromised burden summaries
- **Temporal patterns**: cyclical hour/day/month encodings plus night/evening/weekend indicators

### Modeling Strategy

- **Base models**: 5-fold cross-validated LightGBM baseline, safety-weighted LightGBM, and safety-weighted XGBoost
- **Stacking**: multinomial logistic regression trained on out-of-fold base-model probabilities
- **Safety post-processing**: entropy-triggered one-level conservative shift toward higher acuity when the ensemble is uncertain

### Asymmetric Clinical Cost Matrix

```
                Predicted ->  ESI-1  ESI-2  ESI-3  ESI-4  ESI-5
Actual ESI-1                [  0.0,   1.0,   4.0,  10.0,  20.0 ]
       ESI-2                [  0.5,   0.0,   2.0,   6.0,  12.0 ]
       ESI-3                [  0.3,   0.5,   0.0,   3.0,   8.0 ]
       ESI-4                [  0.2,   0.3,   0.5,   0.0,   3.0 ]
       ESI-5                [  0.1,   0.2,   0.3,   0.5,   0.0 ]
```

Under-triage costs (upper right) are 10-40x higher than over-triage costs (lower left). The code reports this matrix directly during evaluation and uses it to select the conservative-shift threshold.

---

## Datasets

- **Triagegeist Synthetic Dataset** (Laitinen-Fredriksson Foundation) -- 80K train / 20K test records
- Non-commercial research license. See [competition page](https://kaggle.com/competitions/triagegeist) for full terms.

---

## Citation

If you use SafeTriageNet in academic work, please cite:

```
@misc{safetriagenet2026,
  title={SafeTriageNet: Safety-Aware Multimodal Triage with Informative Missingness and Asymmetric Clinical Cost},
  year={2026},
  note={Triagegeist Competition, Laitinen-Fredriksson Foundation}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
