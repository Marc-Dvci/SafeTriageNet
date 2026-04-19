# SafeTriageNet: When Safer Errors Matter More Than Slightly Higher Accuracy

*Safety-aware multimodal triage modeling with informative missingness, lightweight complaint NLP, and asymmetric evaluation*

## Clinical Problem Statement

Emergency triage is a high-pressure ranking problem disguised as a classification problem. A triage nurse is not simply assigning a label; they are deciding who can wait and who cannot. In that setting, all mistakes are not equal. Over-triaging a stable patient is inefficient, but under-triaging a critically ill patient can delay life-saving care.

That distinction shaped our project from the start. We treated triage acuity prediction as a patient-safety problem, not a pure accuracy contest. The goal was not to build the single most aggressive classifier on paper. The goal was to build a model that keeps accuracy competitive while shifting the error profile away from the most dangerous misses.

SafeTriageNet focuses on predicting five-level triage acuity from intake-time information only. We deliberately excluded variables that are effectively post-triage outcomes, such as final disposition and emergency department length of stay. The project therefore stays aligned with the decision point where triage support would actually be used.

## Data

The repository uses the competition-style synthetic Triagegeist dataset:

- `train.csv`: 80,000 visits
- `test.csv`: 20,000 visits
- `chief_complaints.csv`: free-text complaint field for each patient
- `patient_history.csv`: structured comorbidity flags

The modeling target is `triage_acuity` on a 1 to 5 scale. Class imbalance is substantial, with ESI-3 and ESI-4 dominating the training set and ESI-1 representing the rarest, highest-risk group.

One property of the dataset turned out to be especially useful: missing vital signs are not random. Some of that missingness appears to reflect the operational reality of triage rather than simple data quality failure. We tested that directly rather than assuming it, and that analysis became one of the most informative parts of the project.

## Methodology

### 1. Feature engineering grounded in intake-time information

We built a single multimodal feature table that combines:

- structured intake variables such as age, sex, arrival mode, prior utilization, and raw vitals
- derived clinical features such as shock index, mean arterial pressure, pulse pressure, temperature and respiratory abnormality flags, and a count of critical physiologic findings
- complaint-text features derived from lightweight, transparent NLP heuristics
- comorbidity burden summaries from the patient history table
- temporal context features from arrival hour, day, and month

The final modeling table uses 146 features after excluding IDs, raw text, and obvious leakage variables.

### 2. Informative missingness as signal, not just nuisance

Most baseline pipelines would simply impute missing vitals and move on. We took a different approach. For each key vital, we created explicit missingness indicators, a total missing-vital count, and a documentation completeness score.

This mattered. In the audited run, four of six core vitals showed statistically significant acuity-linked missingness patterns, and documentation completeness ranked among the strongest predictive features. Importantly, the effect was not universal: `heart_rate` and `spo2` missingness were not significant in this dataset. We therefore treat missingness as a useful but partial signal, not a universal rule.

### 3. Lightweight complaint NLP

Rather than depend on a heavy transformer stack, we used a reproducible, interpretable text pipeline:

- complaint length and word count
- counts of high-, moderate-, and low-acuity keywords
- targeted flags for high-risk presentations such as chest pain, shortness of breath, altered mental status, trauma, neurologic symptoms, gastrointestinal bleeding, and mental health crisis
- an overall keyword-based acuity signal

This choice kept the notebook fast, transparent, and easy to reproduce in a competition environment while still extracting useful signal from free text.

### 4. Safety-aware model comparison

We trained three base learners with 5-fold stratified cross-validation:

- a baseline LightGBM model
- a safety-weighted LightGBM model
- a safety-weighted XGBoost model

The safety-weighted models use higher sample weights for rare, high-acuity classes. We then stacked the out-of-fold probability predictions from those base models using a multinomial logistic regression meta-learner.

One point is worth stating clearly: the stacked-model evaluation in the final audited version is based on the meta-learner's own out-of-fold predictions, not on in-sample meta-training scores. That makes the reported ensemble performance materially more trustworthy.

### 5. Asymmetric evaluation and conservative post-processing

We defined an asymmetric clinical cost matrix in which under-triage is penalized much more heavily than over-triage, especially when the prediction is off by multiple acuity levels.

We report standard metrics such as accuracy and macro-F1, but we also report:

- under-triage rate
- severe under-triage rate
- over-triage rate
- cost-weighted error
- safety-adjusted kappa

Finally, when the stacked model is uncertain, we apply a one-level conservative shift toward higher acuity. This is intentionally simple. The idea is not to turn every ambiguous patient into a critical alert; it is to make the uncertain cases fail in a safer direction.

## Results

The audited end-to-end run produced the following out-of-fold comparison:

| Model | Accuracy | Macro F1 | Under-Triage Rate | Cost-Weighted Error |
|---|---:|---:|---:|---:|
| Baseline LightGBM | 0.8866 | 0.8964 | 0.0161 | 0.2047 |
| Safety-Weighted LightGBM | 0.8861 | 0.8974 | 0.0137 | 0.2312 |
| Safety-Weighted XGBoost | 0.8841 | 0.8956 | 0.0127 | 0.2349 |
| Stacked Ensemble | 0.8859 | 0.8966 | 0.0112 | 0.2256 |
| SafeTriageNet (Final) | 0.8858 | 0.8965 | 0.0112 | 0.2229 |

The main result is straightforward: the final system lowers under-triage from 1.61% in the baseline LightGBM model to 1.12% while keeping overall accuracy essentially flat. Severe under-triage was 0.0% across all audited models.

There is an important nuance here, and we want to be explicit about it. The baseline LightGBM model retained the lowest raw cost-weighted error. We still selected the final SafeTriageNet configuration because it delivered the best under-triage profile with comparable discrimination and without introducing a large accuracy penalty. In other words, we chose the safer operating point, not the cleanest single-column metric.

That tradeoff is exactly the point of the project. If a triage support model is intended to assist front-line prioritization, the difference between a harmless extra workup and a missed critical patient matters more than a marginal gain in conventional accuracy.

## What We Learned

Three findings stood out.

First, missingness itself is informative. Not every vital behaves the same way, but the pattern of what gets recorded and what does not is part of the triage signal in this dataset. Documentation completeness in particular was one of the strongest predictors.

Second, a broad multimodal feature space helped more than a complicated modeling stack. Some of the most important features came from standard vitals and derived physiologic summaries, but complaint text and history variables added useful context. The project did not need a heavyweight language model to benefit from text.

Third, model selection changes when the objective is safety rather than leaderboard-style optimization. A reasonable baseline already performs well. The value of the final system comes from changing which mistakes are made, not from claiming a dramatic jump in headline accuracy.

## Limitations

This project has several limits that should be stated plainly.

1. The dataset is synthetic. That makes it useful for method development, but it is not a substitute for external validation on real emergency department data.
2. Complaint text was handled with lightweight keyword features. That improves reproducibility, but it does not capture the messiness of real triage language, including abbreviations, misspellings, and institution-specific shorthand.
3. The conservative-shift threshold was selected from out-of-fold meta predictions rather than a fully nested validation design. That is acceptable for this competition setting, but still somewhat optimistic.
4. The code was audited through full pipeline execution rather than a formal unit-test suite. That is enough for a reproducible notebook submission, but it is not the same as production validation.

## Reproducibility

The repository contains:

- a runnable analysis script: `notebooks/safetriagenet.py`
- a Kaggle-friendly notebook export: `notebooks/safetriagenet.ipynb`
- source modules for feature engineering, modeling, and safety evaluation
- regenerated outputs including `submission.csv`, `model_comparison.csv`, figures, and feature importance

The audited pipeline runs end to end in about three minutes on CPU in the current environment. Random seeds are fixed, data-loading paths are now configurable, and the notebook validates that the generated submission matches the expected Kaggle format exactly.

## Closing

SafeTriageNet is not presented as a deployable clinical product. It is a focused, reproducible demonstration of a principle: triage modeling should be judged not only by how often it is right, but by how safely it fails. For this competition, that framing is the real contribution.
