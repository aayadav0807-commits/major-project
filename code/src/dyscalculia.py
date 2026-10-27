from pathlib import Path
import pandas as pd
import os

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "Dyscalculia"

COMPLETE_FILE = os.path.join(DATA_DIR, "Complete_Dataframe_def.xlsx")
ENUMERATION_FILE = os.path.join(DATA_DIR, "Exact_enumeration_task.xlsx")
COMPARISON_FILE = os.path.join(DATA_DIR, "Symbolic_comparison_task.xlsx")
VISUOSPATIAL_FILE = os.path.join(DATA_DIR, "Visuo_spatial_WM_task.xlsx")


def load_dyscalculia_data():

    complete_df = pd.read_excel(COMPLETE_FILE)
    enumeration_df = pd.read_excel(ENUMERATION_FILE)
    comparison_df = pd.read_excel(COMPARISON_FILE)
    visuospatial_df = pd.read_excel(VISUOSPATIAL_FILE)

    return (
        complete_df,
        enumeration_df,
        comparison_df,
        visuospatial_df
    )


def prepare_dyscalculia_dataset():

    complete_df, enumeration_df, comparison_df, visuospatial_df = load_dyscalculia_data()

    print("\n==========================================")
    print("PREPARING DYSCALCULIA DATASET")
    print("==========================================")

    # Convert group names into numbers
    # DD = 1
    # Control = 0

    complete_df["Label"] = complete_df["group"].map({
        "DD": 1,
        "contr": 0
    })

    # Aggregate task-level data for each child

    enumeration_summary = enumeration_df.groupby("Sub").agg({
        "RTs": "mean",
        "Accuracy": "mean"
    }).reset_index()

    enumeration_summary = enumeration_summary.rename(columns={
        "RTs": "Enumeration_RTs",
        "Accuracy": "Enumeration_Accuracy"
    })

    comparison_summary = comparison_df.groupby("Sub").agg({
        "RTs": "mean",
        "Accuracy": "mean"
    }).reset_index()

    comparison_summary = comparison_summary.rename(columns={
        "RTs": "Comparison_RTs",
        "Accuracy": "Comparison_Accuracy"
    })

    visuospatial_summary = visuospatial_df.groupby("Sub").agg({
        "RTs_WM": "mean",
        "Accuracy_WM": "mean"
    }).reset_index()

    visuospatial_summary = visuospatial_summary.rename(columns={
        "RTs_WM": "Visuospatial_RTs",
        "Accuracy_WM": "Visuospatial_Accuracy"
    })

    # Select useful features from complete dataframe

    features = complete_df[
        [
            "Sub",
            "Similarities_IQ",
            "MatrixReas_IQ",
            "Mean_ACC_ANS",
            "Mean_RTs_ANS",
            "Mean_RTs_def",
            "Mean_ACC_13",
            "Mean_ACC_47",
            "MeanRT_13",
            "MeanRT_47",
            "EfficiencyScore_13",
            "EfficiencyScore_47",
            "CoV_13",
            "CoV_47",
            "wm_K",
            "Avanti",
            "Indietro",
            "Average_Corsi",
            "Accuracy_SymbolicComp",
            "RTs_SymbolicComp",
            "Label"
        ]
    ].copy()

    # Merge all task information using participant ID

    dataset = features.merge(
        enumeration_summary,
        on="Sub",
        how="left"
    )

    dataset = dataset.merge(
        comparison_summary,
        on="Sub",
        how="left"
    )

    dataset = dataset.merge(
        visuospatial_summary,
        on="Sub",
        how="left"
    )

    print("\nFinal dataset shape:", dataset.shape)

    print("\nNumber of children:", dataset["Sub"].nunique())

    print("\nClass distribution:")

    print(
        dataset["Label"].value_counts().rename(
            index={
                0: "Control",
                1: "Dyscalculia"
            }
        )
    )

    print("\nMissing values:")

    print(dataset.isnull().sum())

    print("\n==========================================")
    print("DATASET PREPARATION COMPLETE")
    print("==========================================")

    return dataset


if __name__ == "__main__":

    dataset = prepare_dyscalculia_dataset()

    output_file = os.path.join(
        DATA_DIR,
        "dyscalculia_prepared.csv"
    )

    dataset.to_csv(output_file, index=False)

    print("\nPrepared dataset saved to:")

    print(output_file)