import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sps_genai.cnn import SimpleCNN


# Resize CIFAR10 images from 32x32 to 64x64
# to match the CNN architecture required in the assignment
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# Load CIFAR10 training dataset
train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)


# Create model
model = SimpleCNN()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Training
model.train()

num_epochs = 2

for epoch in range(num_epochs):

    running_loss = 0.0

    for images, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    average_loss = running_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {average_loss:.4f}"
    )


print("Training finished.")


# Save trained model
torch.save(
    model.state_dict(),
    "cnn_model.pth"
)

print("Model saved as cnn_model.pth")