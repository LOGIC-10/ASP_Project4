# Auditory Attention Decoding (Project 4)

This repository implements the complete workflow for Project 4 of the Audio Signal Processing course: reconstructing attended speech envelopes from EEG and decoding auditory attention between two competing talkers using the mTRFpy toolbox.

## Directory Structure

```
Project4_Data/        # Subject-specific MATLAB files (S1–S12, train/test splits)
report/               # ACL-style report (`acl_report.md`)
results/              # Generated metrics, JSON logs, and analysis figures
src/aad_pipeline/     # Reusable Python modules for data loading, preprocessing, modelling, evaluation
run_pipeline.py       # Main script: trains subject models, evaluates, and saves results/figures
print_additional_figs.py # Optional script to generate supplementary plots from saved results
```

## Environment Setup

1. Install Anaconda/Miniconda if not already available.
2. Create and activate the project environment (Python ≥ 3.10 recommended):
   ```bash
   conda create -n asp python=3.11 -y
   conda activate asp
   ```
3. Install required Python packages:
   ```bash
   python -m pip install numpy scipy matplotlib scikit-learn mtrf
   ```
   The scripts also rely on standard-library modules (json, csv, pathlib, etc.).

## Running the Pipeline

1. Ensure the dataset folders `Project4_Data/S1` … `Project4_Data/S12` contain the provided `train_data.mat` and `test_data.mat` files.
2. From the project root, execute the full experiment:
   ```bash
   python run_pipeline.py
   ```
   This will:
   - Perform per-subject 5-fold hyperparameter search over ridge regularisation (λ ∈ 10⁻³ … 10³).
   - Train final backward mTRF models, compute reconstruction correlations and attention-decoding accuracies.
   - Evaluate decoding across window sizes (1/5/10/15/30 s).
   - Save summary tables, per-subject CV logs, and required figures under `results/`.

3. (Optional) Generate supplementary diagnostic figures highlighted in the report:
   ```bash
   python print_additional_figs.py
   ```
   This script expects `run_pipeline.py` to have completed so that `results/subject_summary.csv` and related files exist.

## Key Outputs

- `results/subject_summary.csv`: per-subject λ*, reconstruction metrics, and decoding accuracies.
- `results/windowed_accuracy.csv`: window-length accuracies aggregated by subject.
- `results/S*_cv.json`: cross-validation curves for each subject.
- Figures: `reconstruction_example.png`, `aad_window_accuracy.png`, `lambda_cv_curves.png`, `subject_performance_boxplots.png`, `reconstruction_error_spectrum.png`, `decoding_margin_hist.png`, `window_accuracy_subjects.png`.

## Report

The ACL-style report lives at `report/acl_report.md`. Convert to PDF if needed, e.g.:
```bash
pandoc report/acl_report.md -o report/acl_report.pdf
```

## Reproducibility Notes

- The pipeline assumes the EEG and envelope signals are sampled at 64 Hz (as provided). If using different data, adapt the constants in `src/aad_pipeline/data.py` and `run_pipeline.py`.
- All preprocessing normalisation parameters are estimated from training data only and re-used for validation/test splits, aligning with the project specification.
- Random seeds: `run_pipeline.py` fixes NumPy’s seed at 42 for cross-validation shuffling to keep results deterministic.

## Troubleshooting

- **mTRFpy import errors**: verify the package is installed inside the active conda environment (`python -m pip install mtrf`).
- **MATLAB file loading issues**: check the `Project4_Data` hierarchy matches the expected layout and that `.mat` files use the provided structure (`train_struct`/`test_struct`).
- **Plots missing**: re-run `python run_pipeline.py` to regenerate baseline figures before executing `print_additional_figs.py`.

For further analysis or extensions, see the modular code under `src/aad_pipeline/` which exposes utilities for data handling, normalisation, TRF training, and evaluation.
