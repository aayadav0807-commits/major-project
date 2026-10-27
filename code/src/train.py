import os
import torch
import torch.nn as nn
import torch.optim as optim

from dataset import create_dataloaders
from model import DyslexiaCNN


# ==========================================
# CONFIGURATION
# ==========================================

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "dyslexia_cnn.pth")


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("==========================================")
print("DEVICE:", device)
print("==========================================")


# ==========================================
# CREATE MODEL DIRECTORY
# ==========================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

print("\nLoading dataset...")

train_loader, test_loader = create_dataloaders(
    batch_size=BATCH_SIZE
)

print("Dataset loaded successfully!")


# ==========================================
# CREATE MODEL
# ==========================================

model = DyslexiaCNN(num_classes=3)

model = model.to(device)


# ==========================================
# LOSS FUNCTION
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# OPTIMIZER
# ==========================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

print("\n==========================================")
print("STARTING TRAINING")
print("==========================================")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch + 1}/{EPOCHS}")

    for batch_index, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (predicted == labels).sum().item()

        # Show progress every 100 batches
        if (batch_index + 1) % 100 == 0:

            current_accuracy = 100 * correct / total

            print(
                f"Batch [{batch_index + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                f"Accuracy: {current_accuracy:.2f}%"
            )

    epoch_loss = running_loss / len(train_loader)

    epoch_accuracy = 100 * correct / total

    print("\n------------------------------------------")
    print(f"Epoch Loss: {epoch_loss:.4f}")
    print(f"Epoch Accuracy: {epoch_accuracy:.2f}%")
    print("------------------------------------------")


# ==========================================
# SAVE MODEL
# ==========================================

torch.save(model.state_dict(), MODEL_PATH)

print("\n==========================================")
print("TRAINING COMPLETED")
print("==========================================")

print("Model saved at:")
print(MODEL_PATH)