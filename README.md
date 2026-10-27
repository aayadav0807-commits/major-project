# NeuroScreen AI (research prototype)

NeuroScreen AI is a Streamlit research/educational prototype for recording child handwriting, reading, mathematics-task and experimental eye-tracking measurements. It is **not a clinical diagnostic system**. Demo thresholds are review/recording-quality prompts only, not validated diagnostic cut-offs.

## Recommended environment

Use **64-bit Python 3.11**. From the repository root in PowerShell:

```powershell
uv python install 3.11
uv venv --python 3.11
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r code/requirements.txt
streamlit run code/app_hybrid_test.py
```

On first run, the app creates a local database at `code/neuroscreen.db`. Create the first administrator account in the UI. The database is ignored by Git and must never be committed.

## Local datasets

Datasets are not stored in this public repository. Place legally obtained datasets under `code/data/` using the exact folder names expected by each script, including `Gambo`, `Dyscalculia`, `DyslexiaScreening`, and `Eye Tracking/ETDD70`. Keep datasets containing personal or identifiable child information out of Git.

## Model boundaries

- The active handwriting classifier loads `code/models/dyslexia_cnn.pth` and predicts dataset classes `Corrected`, `Normal`, and `Reversal`. These are dataset labels, not diagnoses.
- The active dyscalculia inference pipeline expects the 11 app-collected features in `DYSCALCULIA_FEATURES`. Train with `code/src/train_dyscalculia_app.py`, not the older all-feature script.
- `eye_tracking_model.joblib` and `dyslexia_screening_*.pth` are separate research artifacts, not loaded by the live app. Do not claim the app uses them unless a tested integration is implemented.
- Eye tracking currently measures calibration/tracking quality and target error. It does not output an ML-based disability prediction.
- The dyslexia-screening trainer uses an image-level split without participant IDs. If multiple images belong to the same child, the reported test score can be optimistic. Use participant-level/group splitting for a credible evaluation.
- `code/src/assessment_thresholds.py` contains demo review/quality gates only, not clinical cut-offs.

## Privacy and source control

Local databases, datasets, environment files, and Python bytecode are ignored by `.gitignore`. If a database with real personal data was previously committed, deleting it in a later commit does not remove it from Git history. Coordinate a history rewrite and rotate exposed credentials/secrets. Never publish child names, records, or password hashes.

## Limitations

This repository cannot be fully validated from source alone: local datasets are absent, and model quality requires reproducible evaluation on held-out participants. Test with synthetic/demo records first and obtain appropriate consent before real-world child assessment.
