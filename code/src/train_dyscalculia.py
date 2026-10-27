import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "Dyscalculia" / "dyscalculia_prepared.csv"


print("==========================================")
print("DYSCALCULIA AI MODEL TRAINING")
print("==========================================")

# Load prepared dataset
df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully!")
print("Total children:", len(df))

# Remove participant ID
X = df.drop(columns=["Sub", "Label"])

# Target
y = df["Label"]

print("\nFeatures:", X.shape[1])
print("Children with labels:")
print(y.value_counts())

# Handle missing values + scaling + Random Forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

# 5-fold cross validation
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="accuracy"
)

print("\n==========================================")
print("5-FOLD CROSS VALIDATION")
print("==========================================")

for i, score in enumerate(scores, 1):
    print(f"Fold {i} Accuracy: {score * 100:.2f}%")

print(f"\nAverage Accuracy: {scores.mean() * 100:.2f}%")
print(f"Standard Deviation: {scores.std() * 100:.2f}%")

# Train final model using all available children
model.fit(X, y)

print("\n==========================================")
print("FINAL MODEL TRAINED")
print("==========================================")

print("Training completed successfully!")

# Save model
import joblib

MODEL_FILE = "models/dyscalculia_model.pkl"

joblib.dump(model, MODEL_FILE)

print("\nModel saved to:")
print(MODEL_FILE)

print("\n==========================================")
print("DYSCALCULIA TRAINING COMPLETE")
print("==========================================")