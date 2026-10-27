from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ==========================================
# DATASET PATHS
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "Gambo" / "Train"
TEST_DIR = ROOT / "data" / "Gambo" / "Test"


# ==========================================
# IMAGE TRANSFORMATIONS
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# ==========================================
# LOAD DATASETS
# ==========================================

def load_datasets():

    train_dataset = datasets.ImageFolder(
        root=TRAIN_DIR,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=TEST_DIR,
        transform=transform
    )

    return train_dataset, test_dataset


# ==========================================
# CREATE DATA LOADERS
# ==========================================

def create_dataloaders(batch_size=32):

    train_dataset, test_dataset = load_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, test_loader


# ==========================================
# TEST DATASET
# ==========================================

if __name__ == "__main__":

    train_dataset, test_dataset = load_datasets()

    print("==========================================")
    print("DATASET INFORMATION")
    print("==========================================")

    print("Classes:", train_dataset.classes)

    print("Class to index:", train_dataset.class_to_idx)

    print("Training images:", len(train_dataset))

    print("Testing images:", len(test_dataset))

    print("==========================================")
    print("Dataset loaded successfully!")