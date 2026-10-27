import os

# Dataset location
dataset_path = "data/Gambo"

# Classes
classes = ["Normal", "Corrected", "Reversal"]

print("\n========== TRAIN DATA ==========")

for class_name in classes:
    folder = os.path.join(dataset_path, "Train", class_name)

    if os.path.exists(folder):
        images = os.listdir(folder)
        print(f"{class_name}: {len(images)} images")
    else:
        print(f"{class_name}: Folder not found!")

print("\n========== TEST DATA ==========")

for class_name in classes:
    folder = os.path.join(dataset_path, "Test", class_name)

    if os.path.exists(folder):
        images = os.listdir(folder)
        print(f"{class_name}: {len(images)} images")
    else:
        print(f"{class_name}: Folder not found!")

print("\nDataset check completed!")