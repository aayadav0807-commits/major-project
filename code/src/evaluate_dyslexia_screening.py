
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT / "data" / "DyslexiaScreening"
    / "Complete And Balance Hand-written Dataset" / "Dataset"
)

MODEL_PATH = ROOT / "models" / "dyslexia_screening_letterbox_cnn_v2.pth"

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 16

# --------------------------------------------------
# SAME PREPROCESSING AS TRAINING
# --------------------------------------------------
class LetterboxTransform:
    def __call__(self, image):
        image = ImageOps.grayscale(image)
        image.thumbnail(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        canvas = Image.new(
            "L", (IMAGE_SIZE, IMAGE_SIZE), color=255
        )
        left = (IMAGE_SIZE - image.width) // 2
        top = (IMAGE_SIZE - image.height) // 2
        canvas.paste(image, (left, top))

        return canvas


transform = transforms.Compose([
    LetterboxTransform(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --------------------------------------------------
# CNN ARCHITECTURE — MUST MATCH TRAINING
# --------------------------------------------------
class DyslexiaScreeningCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# --------------------------------------------------
# CHECK FILES
# --------------------------------------------------
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

if not (DATA_DIR / "No").is_dir() or not (DATA_DIR / "Yes").is_dir():
    raise FileNotFoundError(
        f"Dataset folders not found: {DATA_DIR}"
    )

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

checkpoint = torch.load(
    MODEL_PATH, map_location="cpu", weights_only=True
)

saved_classes = checkpoint["classes"]

if dataset.classes != saved_classes:
    raise ValueError(
        f"Class mismatch. Model: {saved_classes}; "
        f"Dataset: {dataset.classes}"
    )

# Recreate the same seeded split used during training.
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

_, test_data = random_split(
    dataset,
    [train_size, test_size],
    generator=torch.Generator().manual_seed(SEED)
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
model = DyslexiaScreeningCNN(
    num_classes=len(saved_classes)
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(predictions.tolist())

# --------------------------------------------------
# RESULTS
# --------------------------------------------------
print("\n=== LETTERBOX CNN EVALUATION ===")
print("Model:", MODEL_PATH.name)
print("Total dataset images:", len(dataset))
print("Training split images:", train_size)
print("Test split images:", len(test_data))
print(f"Accuracy: {accuracy_score(y_true, y_pred):.2%}")
print(
    "Balanced accuracy:",
    f"{balanced_accuracy_score(y_true, y_pred):.2%}"
)

print("\nClassification report:")
print(classification_report(
    y_true,
    y_pred,
    labels=list(range(len(saved_classes))),
    target_names=saved_classes,
    zero_division=0,
    digits=3
))

print("\nConfusion matrix:")
print("Rows = actual classes; columns = predicted classes")
print("Class order:", saved_classes)
print(confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(saved_classes)))
))

print("\n=== LIMITATIONS ===")
print("- Evaluation uses an image-level split.")
print("- Participant-level separation is not verified.")
print("- Results do not establish clinical diagnostic accuracy.")
