import torch
from sklearn.metrics import classification_report, confusion_matrix
from dataset import create_dataloaders
from model import DyslexiaCNN


# ==========================================
# SETTINGS
# ==========================================

BATCH_SIZE = 32
MODEL_PATH = "models/dyslexia_cnn.pth"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("MODEL EVALUATION")
print("==========================================")

print("Device:", device)


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading test dataset...")

train_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE
)

print("Test images:", len(test_loader.dataset))


# ==========================================
# LOAD MODEL
# ==========================================

print("\nLoading trained model...")

model = DyslexiaCNN(num_classes=3)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device)

model.eval()

print("Model loaded successfully!")


# ==========================================
# EVALUATION
# ==========================================

all_predictions = []
all_labels = []

correct = 0
total = 0

print("\n==========================================")
print("EVALUATING MODEL")
print("==========================================")

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        all_predictions.extend(
            predicted.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ==========================================
# TEST ACCURACY
# ==========================================

accuracy = 100 * correct / total

print("\n==========================================")
print("TEST RESULTS")
print("==========================================")

print(f"Correct predictions : {correct}")
print(f"Total test images   : {total}")
print(f"Test Accuracy       : {accuracy:.2f}%")


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

class_names = test_loader.dataset.classes

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=class_names
    )
)


# ==========================================
# CONFUSION MATRIX
# ==========================================

print("\n==========================================")
print("CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print(cm)

print("\n==========================================")
print("EVALUATION COMPLETED")
print("==========================================")