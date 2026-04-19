# Submission Audit

Audit date: 2026-04-14 (updated)

## Status

The project is ready for Kaggle submission. The pipeline runs end to end, produces a valid `submission.csv`, ships a self-contained Kaggle notebook (no `src/` dependency, no `__file__`), and has a compliant cover image. The final Kaggle writeup is drafted in [KAGGLE_WRITEUP_V1.md](./KAGGLE_WRITEUP_V1.md) at ~1965 words (under the 2000-word limit).

## Issues Found And Fixed

1. **Categorical encoding was not stable between train and test**

   The original pipeline encoded categorical columns independently in train and test. That is safe only when both splits happen to expose the exact same category set in the exact same order. In a competition setting, that is an unnecessary inference risk.

   Fix:
   - Added explicit training-set category maps in [src/features.py](../src/features.py)
   - Applied the same fitted mappings to test-time encoding in [notebooks/safetriagenet.py](../notebooks/safetriagenet.py)

2. **Stacked ensemble evaluation was optimistic**

   The first version trained the meta-learner on the full out-of-fold base-model predictions and then evaluated it on those same meta-features. That inflates the reported stacked metrics.

   Fix:
   - Reworked [src/models.py](../src/models.py) so the meta-learner now produces its own 5-fold out-of-fold predictions for honest evaluation
   - Updated [notebooks/safetriagenet.py](../notebooks/safetriagenet.py) to report stacked metrics from those out-of-fold meta predictions while still fitting a final full-data meta-model for test inference

3. **Data loading was too brittle**

   The notebook assumed a single hard-coded relative path to the CSV files.

   Fix:
   - Added `TRIAGEGEIST_DATA_DIR` support and fallback path discovery in [notebooks/safetriagenet.py](../notebooks/safetriagenet.py)

4. **Documentation and claims were out of sync with the code**

   Several statements in the README and notebook were stronger than the audited implementation justified. Examples:
   - old metrics no longer matched the regenerated outputs
   - the stacker was described in a way that implied a more optimistic evaluation than was actually valid
   - the missingness narrative overstated how universal the signal was across vitals

   Fix:
   - Updated [README.md](../README.md)
   - Tightened explanatory text in [notebooks/safetriagenet.py](../notebooks/safetriagenet.py)

5. **Cover image did not meet Kaggle size requirements**

   The existing asset was `458x239`. The competition requirement is `560x280`.

   Fix:
   - Resized and repadded [assets/cover_image.png](../assets/cover_image.png) to `560x280`

6. **The repository did not include an actual notebook file**

   The repo had an executable Python script, but not a `.ipynb` artifact that could be uploaded directly or browsed as a notebook.

   Fix:
   - Added [notebooks/safetriagenet.ipynb](../notebooks/safetriagenet.ipynb) as a notebook export of the audited pipeline

7. **First notebook export would crash on Kaggle**

   The initial notebook export used `__file__` (undefined in Jupyter cells) and imported `from features import ...` / `from safety import ...` — both of which fail on Kaggle where the `src/` directory is not present.

   Fix:
   - Added [build/build_kaggle_notebook.py](../build/build_kaggle_notebook.py), which generates a fully self-contained notebook by inlining the `src/` modules as code cells (with top-level imports and module docstrings stripped via `ast`)
   - Rewrote data-path discovery to try `TRIAGEGEIST_DATA_DIR`, then `/kaggle/input/triagegeist`, then a sweep over `/kaggle/input/*`, then local fallbacks
   - Verified the new [notebooks/safetriagenet.ipynb](../notebooks/safetriagenet.ipynb) by flattening it back to a Python script and running it end to end — all 24 code cells parse and produce the same metrics and submission as the legacy script

8. **Feature-importance figure had a color-mapping bug**

   In [notebooks/safetriagenet.py](../notebooks/safetriagenet.py), `bars[top_n-1-i]` was used inside the per-feature coloring loop. Because bar `i` corresponds to feature row `i` (both in descending-importance order), this inverted the color assignment: missingness bars got the "other" color and vice versa.

   Fix:
   - Changed the loop to use `bars[i]` so colors align with the iterated row
   - The Kaggle notebook version uses `color=feature_colors` inline on the `barh` call, which sidesteps the indexing entirely
   - Regenerated `outputs/fig9_feature_importance.png` with the corrected colors

## Verification Performed

- `python -m compileall src notebooks`
  Result: passed

- `python -u notebooks/safetriagenet.py`
  Result: passed end to end in about 160 seconds on the current machine

- `python build/build_kaggle_notebook.py` and parse-check of every code cell via `ast.parse`
  Result: 37 cells written (13 markdown, 24 code); all code cells parse cleanly

- Flattened-notebook execution (notebook -> .py -> run)
  Result: reproduces the audited metrics exactly (Acc 0.8858, UTR 0.0112, CWE 0.2229) and the same 20000 x 2 submission

- Submission format validation inside the notebook
  Result:
  - shape matched the sample submission: `20000 x 2`
  - columns matched exactly: `patient_id`, `triage_acuity`

- Cover image check
  Result: `assets/cover_image.png` is `560x280`

## Regenerated Metrics

From the audited end-to-end run:

| Model | Accuracy | Macro F1 | Under-Triage Rate | Cost-Weighted Error |
|---|---:|---:|---:|---:|
| Baseline LightGBM | 0.8866 | 0.8964 | 0.0161 | 0.2047 |
| Safety-Weighted LightGBM | 0.8861 | 0.8974 | 0.0137 | 0.2312 |
| Safety-Weighted XGBoost | 0.8841 | 0.8956 | 0.0127 | 0.2349 |
| Stacked Ensemble | 0.8859 | 0.8966 | 0.0112 | 0.2256 |
| SafeTriageNet (Final) | 0.8858 | 0.8965 | 0.0112 | 0.2229 |

Interpretation:
- The final submission materially improves under-triage versus the baseline while holding accuracy roughly flat.
- The baseline LightGBM still has the lowest raw cost-weighted error, so the final model should be described as the **safer** choice rather than the universally best choice on every metric.

## Remaining Risks And Caveats

1. **Synthetic-data limitation**

   The entire analysis is performed on synthetic data. That is acceptable for the competition framing, but it must be stated clearly in the writeup.

2. **Missingness is informative, but not uniformly so**

   In the audited run, four of six core vitals showed significant acuity-linked missingness. `heart_rate` and `spo2` did not. The writeup should describe missingness as an important signal in this dataset, not as a universal property of every vital.

3. **Conservative-shift threshold selection is still mildly optimistic**

   The entropy threshold is selected from a small sweep on out-of-fold meta probabilities and then reported on those same out-of-fold probabilities. That is acceptable for a competition notebook, but it is not the same as a fully nested estimate.

4. **No unit-test suite**

   There is still no formal automated test suite. For this project, the strongest verification is the successful full pipeline execution. That is enough for submission, but not equivalent to production-grade testing.

## Recommended Submission Package

- Public Kaggle notebook: [notebooks/safetriagenet.ipynb](../notebooks/safetriagenet.ipynb) (self-contained; runs on Kaggle with data at `/kaggle/input/triagegeist/`)
- Public repository or project link: this `SafeTriageNet` directory with the updated README
- **Final Kaggle writeup**: [docs/KAGGLE_WRITEUP_V1.md](./KAGGLE_WRITEUP_V1.md) (1965 words, within the 2000-word limit; V0 retained as historical draft)
- Cover image: [assets/cover_image.png](../assets/cover_image.png)

## Bottom Line

The main correctness and credibility issues uncovered during audit have been fixed. The project is now submission-ready provided the Kaggle writeup is uploaded, the notebook is made public, and the final submission uses the regenerated artifacts from this audited version.
