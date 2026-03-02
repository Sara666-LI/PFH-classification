# PFH-classification

## End-to-end binary classification pipeline

This repository now includes a complete script for PFH binary classification using:

- **Clinical model (MClinic)**: `Age`, `AFP` (and fallback to available clinical columns if needed).
- **Radiomics model (MRad)**.
- **Foundation model (MFound)**.
- **Feature-level fusion models**:
  - `MClinic-Rad`
  - `MRad-Found`
  - `MClinic-Rad-Found`
- **Late fusion models** (probability averaging across unimodal models):
  - `LateFusion_MClinic-Rad`
  - `LateFusion_MRad-Found`
  - `LateFusion_MClinic-Rad-Found`

Implemented methods include:

- SMOTE in training folds.
- Repeated stratified cross-validation.
- Cox-style (univariate p-value) feature filtering for radiomics/foundation.
- Bootstrap feature selection.
- SHAP explanations.
- ROC plots (per-model + overall comparison).
- Boxplots for AUC / Balanced Accuracy / Sensitivity / Specificity.
- Radar plot for mean validation metrics.
- Scatter plot of clinical vs Rad-Found risk score with PFH label overlay.
- Complete fold-level training/validation metrics and probabilities saved to CSV.

## Script

- `pfh_binary_classification_pipeline.py`

## Example usage

```bash
python pfh_binary_classification_pipeline.py \
  --radiomics-file df_radiomics_aligned.xlsx \
  --foundation-file df_found_aligned.xlsx \
  --target-col PFH \
  --id-col ID \
  --output-dir outputs \
  --n-splits 5 \
  --n-repeats 5
```

## Main outputs (in `--output-dir`)

- Per-model fold metrics and probabilities:
  - `<MODEL>_fold_metrics.csv`
  - `<MODEL>_fold_predictions.csv`
- Combined files:
  - `all_models_fold_metrics.csv`
  - `all_models_fold_predictions.csv`
- Figures:
  - `roc_<MODEL>.png`
  - `roc_all_models_comparison.png`
  - `metrics_boxplots_all_models.png`
  - `radar_metrics.png`
  - `scatter_clinical_vs_rad_found_risk.png`
  - `shap_summary_<MODEL>.png`
- Manifest:
  - `feature_and_model_manifest.json`
