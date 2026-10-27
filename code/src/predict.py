import torch
from PIL import Image
from torchvision import transforms

from model import DyslexiaCNN


# ==========================================
# SETTINGS
# ==========================================

MODEL_PATH = "models/dyslexia_cnn.pth"

IMAGE_PATH = "data/Gambo/Test/Normal/Normal2177 (2).png"


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("==========================================")
print("HANDWRITING PREDICTION")
print("==========================================")

print("Device:", device)


# ==========================================
# IMAGE TRANSFORMATION
# ==========================================

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


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
# LOAD IMAGE
# ==========================================

print("\nLoading handwriting image...")

image = Image.open(IMAGE_PATH)

print("Image loaded successfully!")
print("Image:", IMAGE_PATH)


# ==========================================
# PREPROCESS IMAGE
# ==========================================

image_tensor = transform(image)

image_tensor = image_tensor.unsqueeze(0)

image_tensor = image_tensor.to(device)


# ==========================================
# PREDICTION
# ==========================================

print("\nMaking prediction...")

with torch.no_grad():

    output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    ).item()


# ==========================================
# CLASS NAMES
# ==========================================

classes = [
    "Corrected",
    "Normal",
    "Reversal"
]


prediction = classes[predicted_class]

confidence = (
    probabilities[0][predicted_class].item()
    * 100
)


# ==========================================
# RESULT
# ==========================================

print("\n==========================================")
print("PREDICTION RESULT")
print("==========================================")

print("Predicted class :", prediction)
print(f"Confidence      : {confidence:.2f}%")

print("\nClass probabilities:")

for i, class_name in enumerate(classes):

    probability = (
        probabilities[0][i].item()
        * 100
    )

    print(
        f"{class_name:10s}: "
        f"{probability:.2f}%"
    )

print("\n==========================================")
print("PREDICTION COMPLETED")
print("==========================================")