
from pathlib import Path

import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# PATHS
# --------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / "data" / "Eye Tracking" / "ETDD70" / "data"
LABEL_FILE = (
    BASE / "data" / "Eye Tracking" / "ETDD70"
    / "dyslexia_class_label.csv"
)

# Trial-level numerical features.
FEATURES = [
    "n_fix_trial",
    "mean_fix_dur_trial",
    "n_sacc_trial",
    "mean_sacc_dur_trial",
    "mean_sacc_ampl_trial",
    "ratio_progress_regress_trial",
    "n_regress_trial",
    "dwell_time_trial",
]

TASKS = [
    "T1_Syllables",
    "T4_Meaningful_Text",
    "T5_Pseudo_Text",
]


def main():
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"Labels not found: {LABEL_FILE}")

    files = list(DATA_DIR.glob("*_metrics.csv"))
    if not files:
        raise FileNotFoundError(f"No metrics CSV files in {DATA_DIR}")

    frames = []
    for file in files:
        df = pd.read_csv(file)

        required = {"sid", "task", "trialid", *FEATURES}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{file.name} missing columns: {missing}")

        # Keep trial-level features, not repeated AOI-level rows.
        df = df[["sid", "task", "trialid", *FEATURES]].copy()
        df["sid"] = df["sid"].astype(str)
        df = df.drop_duplicates(
            subset=["sid", "task", "trialid"]
        )
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric, errors="coerce"
    )

    # One row per participant and task.
    grouped = data.groupby(["sid", "task"])[FEATURES].mean()

    # One row per participant; separate features by task.
    wide = grouped.unstack("task")
    wide.columns = [
        f"{feature}__{task}"
        for feature, task in wide.columns
    ]
    wide.index = wide.index.astype(str)

    labels = pd.read_csv(LABEL_FILE)
    labels["subject_id"] = labels["subject_id"].astype(str)
    labels = labels.drop_duplicates("subject_id")

    label_map = {
        "non-dyslexic": 0,
        "dyslexic": 1,
    }
    labels["target"] = labels["label"].map(label_map)

    if labels["target"].isna().any():
        raise ValueError("Unknown or missing class labels found.")

    dataset = wide.join(
        labels.set_index("subject_id")["target"],
        how="inner",
    ).dropna(subset=["target"])

    X = dataset.drop(columns="target")
    y = dataset["target"].astype(int)

    # Remove features that are entirely missing.
    X = X.dropna(axis=1, how="all")

    print("Participants:", len(dataset))
    print("Features:", X.shape[1])
    print("Class counts (0=non-dyslexic, 1=dyslexic):")
    print(y.value_counts().sort_index())

    if len(dataset) < 10 or y.nunique() != 2:
        raise ValueError("Not enough matched participants or classes.")

    # Every participant appears only once in X, preventing
    # the same participant from leaking across CV folds.
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
    )

    predictions = cross_val_predict(model, X, y, cv=cv)

    print("\n--- 5-FOLD CROSS-VALIDATION RESULTS ---")
    print(f"Accuracy:  {accuracy_score(y, predictions):.3f}")
    print(f"Precision: {precision_score(y, predictions, zero_division=0):.3f}")
    print(f"Recall:    {recall_score(y, predictions, zero_division=0):.3f}")
    print(f"F1-score:  {f1_score(y, predictions, zero_division=0):.3f}")

    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(y, predictions, labels=[0, 1]))

    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=[0, 1],
            target_names=["non-dyslexic", "dyslexic"],
            zero_division=0,
        )
    )

    # Train the final pipeline on all labelled participants
    model.fit(X, y)

    # Save model and feature information
    model_dir = BASE / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "eye_tracking_model.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_columns": list(X.columns),
            "label_map": {
                0: "non-dyslexic",
                1: "dyslexic",
            },
            "training_participants": len(dataset),
            "tasks": TASKS,
        },
        model_path,
    )

    print(f"\nSaved trained model to: {model_path}")


if __name__ == "__main__":
    main()
