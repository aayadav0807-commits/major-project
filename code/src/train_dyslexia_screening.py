
from pathlib import Path
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
from PIL import Image, ImageOps

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    ROOT / "data" / "DyslexiaScreening"
    / "Complete And Balance Hand-written Dataset" / "Dataset"
)

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "dyslexia_screening_letterbox_cnn_v2.pth"

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------
# CHECK DATASET
# --------------------------------------------------
if not (DATA_DIR / "Yes").is_dir() or not (DATA_DIR / "No").is_dir():
    raise FileNotFoundError(f"Dataset folders not found: {DATA_DIR}")

# --------------------------------------------------
# IMAGE PREPROCESSING — ASPECT-RATIO-PRESERVING LETTERBOX
# --------------------------------------------------
IMAGE_SIZE = 128


class LetterboxTransform:
    def __call__(self, image):
        image = ImageOps.grayscale(image)

        # Fit the complete image inside 128 x 128.
        image.thumbnail(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        # Center it on a white square without cropping.
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

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

print("Dataset:", DATA_DIR)
print("Classes:", dataset.classes)
print("Images:", len(dataset))

if len(dataset) == 0 or set(dataset.classes) != {"No", "Yes"}:
    raise ValueError("Expected non-empty No and Yes folders.")

# --------------------------------------------------
# TRAIN / TEST SPLIT
# Preferred: participant_manifest.csv with columns
# relative_path,participant_id; paths are relative to DATA_DIR.
# --------------------------------------------------
MANIFEST_PATH = DATA_DIR / "participant_manifest.csv"
if MANIFEST_PATH.is_file():
    manifest = pd.read_csv(MANIFEST_PATH)
    if not {"relative_path", "participant_id"}.issubset(manifest.columns):
        raise ValueError("participant_manifest.csv needs relative_path,participant_id columns.")
    manifest["relative_path"] = manifest["relative_path"].astype(str).str.replace("\\\\", "/", regex=False)
    participant_by_path = dict(zip(manifest["relative_path"], manifest["participant_id"]))
    sample_relpaths = [
        Path(sample_path).resolve().relative_to(DATA_DIR.resolve()).as_posix()
        for sample_path, _ in dataset.samples
    ]
    missing = [relpath for relpath in sample_relpaths if relpath not in participant_by_path]
    if missing:
        raise ValueError(f"Manifest is missing {len(missing)} image(s), e.g. {missing[0]}")
    groups = [participant_by_path[relpath] for relpath in sample_relpaths]
    if len(set(groups)) < 2:
        raise ValueError("At least two different participant IDs are required.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_indices, test_indices = next(splitter.split(range(len(dataset)), dataset.targets, groups))
    if set(groups[i] for i in train_indices) & set(groups[i] for i in test_indices):
        raise RuntimeError("Participant leakage detected in train/test split.")
    train_data = Subset(dataset, train_indices.tolist())
    test_data = Subset(dataset, test_indices.tolist())
    PARTICIPANT_SPLIT = True
    print("Using participant-level split:", MANIFEST_PATH)
else:
    # Fallback keeps demo training runnable but does not establish independent-child accuracy.
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(
        dataset, [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    PARTICIPANT_SPLIT = False
    print("WARNING: participant_manifest.csv not found.")
    print("Image-level split only; test accuracy may be inflated.")

train_loader = DataLoader(
    train_data, batch_size=16, shuffle=True, num_workers=0
)
test_loader = DataLoader(
    test_data, batch_size=16, shuffle=False, num_workers=0
)

# --------------------------------------------------
# CNN MODEL
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


device = torch.device("cpu")
model = DyslexiaScreeningCNN(
    num_classes=len(dataset.classes)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
EPOCHS = 10

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {running_loss / total:.4f} | "
        f"Training accuracy: {100 * correct / total:.2f}%"
    )

# --------------------------------------------------
# PRELIMINARY HELD-OUT IMAGE EVALUATION
# --------------------------------------------------
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

test_accuracy = 100 * correct / total

print(f"\nTest accuracy: {test_accuracy:.2f}%")
if not PARTICIPANT_SPLIT:
    print("WARNING: image-level metric only; participant leakage is possible.")
else:
    print("Participant-level split used; external validation is still required.")
print("This result does not establish clinical diagnostic accuracy.")

# --------------------------------------------------
# SAVE TO A NEW FILE — NEVER OVERWRITE
# --------------------------------------------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)

if MODEL_PATH.exists():
    raise FileExistsError(
        f"{MODEL_PATH} already exists. "
        "Choose another filename before retraining."
    )

torch.save({
    "model_state_dict": model.state_dict(),
    "classes": dataset.classes,
    "image_size": IMAGE_SIZE,
    "normalization_mean": 0.5,
    "normalization_std": 0.5,
    "preprocessing": "grayscale_aspect_ratio_preserving_letterbox",
    "seed": SEED,
    "epochs": EPOCHS,
    "test_accuracy": test_accuracy,
    "participant_level_split": PARTICIPANT_SPLIT,
    "evaluation_note": (
        "Participant-level split" if PARTICIPANT_SPLIT else
        "Image-level split only; possible participant leakage. Do not report as independent-child accuracy."
    )
}, MODEL_PATH)

print("New model saved to:", MODEL_PATH)
print("Existing Gambo model was not changed.")
