import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "data/Dyscalculia/dyscalculia_prepared.csv"

df = pd.read_csv(DATA_PATH)

print("\n============================================================")
print("DYScalculia APP MODEL TRAINING")
print("============================================================")

print("\nDataset shape:", df.shape)


# ============================================================
# 2. FEATURES COLLECTED BY OUR APP
# ============================================================

features = [
    "Mean_ACC_ANS",
    "Mean_RTs_ANS",
    "Accuracy_SymbolicComp",
    "RTs_SymbolicComp",
    "Enumeration_RTs",
    "Enumeration_Accuracy",
    "wm_K",
    "Comparison_RTs",
    "Comparison_Accuracy",
    "Visuospatial_RTs",
    "Visuospatial_Accuracy"
]


# ============================================================
# 3. CHECK FEATURES
# ============================================================

print("\nChecking required features...")

missing_features = [
    feature for feature in features
    if feature not in df.columns
]

if missing_features:

    print("\nERROR: Missing features:")
    print(missing_features)

    raise ValueError(
        "Some required features are missing from the dataset."
    )

print("All required features found.")


# ============================================================
# 4. PREPARE X AND Y
# ============================================================

X = df[features]
y = df["Label"]


print("\nFeatures used by the model:")
for feature in features:
    print("-", feature)

print("\nTarget distribution:")
print(y.value_counts())


# ============================================================
# 5. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 6. CREATE MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            )
        )
    ]
)


# ============================================================
# 7. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed successfully.")


# ============================================================
# 8. TEST MODEL
# ============================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n============================================================")
print("MODEL EVALUATION")
print("============================================================")

print(
    f"\nTest Accuracy: {accuracy * 100:.2f}%"
)


print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

MODEL_PATH = "models/dyscalculia_app_model.pkl"

joblib.dump(
    model,
    MODEL_PATH
)


print("\n============================================================")
print("MODEL SAVED")
print("============================================================")

print(
    f"\nModel saved to:\n{MODEL_PATH}"
)

print("\nFeatures stored in model:")

for feature in features:
    print("-", feature)

print("\n============================================================")
print("TRAINING COMPLETE")
print("============================================================")