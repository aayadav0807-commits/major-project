import torch
import torch.nn as nn


class DyslexiaCNN(nn.Module):

    def __init__(self, num_classes=3):
        super(DyslexiaCNN, self).__init__()

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 8 * 8, 256),
            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================
# TEST MODEL
# =========================

if __name__ == "__main__":

    model = DyslexiaCNN(num_classes=3)

    print("========== MODEL INFORMATION ==========")

    print(model)

    # Test with one batch of images
    test_input = torch.randn(4, 1, 128, 128)

    output = model(test_input)

    print("\nInput shape :", test_input.shape)
    print("Output shape:", output.shape)

    print("\nModel created successfully!")