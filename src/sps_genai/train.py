import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sps_genai.cnn import SimpleCNN


# CIFAR-10 classes
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


# Convert images to tensors
transform = transforms.Compose([
    transforms.ToTensor()
])


# Load CIFAR-10 training data
train_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)


# Create the CNN
model = SimpleCNN()

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the model
EPOCHS = 2

for epoch in range(EPOCHS):
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{EPOCHS}, "
        f"Loss: {running_loss / len(train_loader):.4f}"
    )


# Save trained model
torch.save(model.state_dict(), "cnn_model.pth")

print("Training finished.")
print("Model saved as cnn_model.pth")
