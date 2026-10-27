import pandas as pd
import joblib

MODEL_FILE = "models/dyscalculia_model.pkl"
DATA_FILE = "data/Dyscalculia/dyscalculia_prepared.csv"

print("==========================================")
print("DYSCALCULIA AI PREDICTION TEST")
print("==========================================")

# Load model
model = joblib.load(MODEL_FILE)

# Load dataset
df = pd.read_csv(DATA_FILE)

# Keep participant IDs separately
participant_ids = df["Sub"]

# Remove ID and target label
X = df.drop(columns=["Sub", "Label"])

# Make predictions
predictions = model.predict(X)

# Get prediction probabilities
probabilities = model.predict_proba(X)

print("\nPrediction results:\n")

for i in range(len(df)):
    if predictions[i] == 1:
        result = "Dyscalculia"
    else:
        result = "Control"

    confidence = probabilities[i].max() * 100

    print(
        f"Child {participant_ids.iloc[i]} "
        f"-> {result} "
        f"({confidence:.2f}% confidence)"
    )

print("\n==========================================")
print("PREDICTION TEST COMPLETE")
print("==========================================")