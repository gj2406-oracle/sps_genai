import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sps_genai.cnn import SimpleCNN


# Resize test images to 64x64
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


# Load CIFAR10 test dataset
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)


# Load model
model = SimpleCNN()

model.load_state_dict(
    torch.load(
        "cnn_model.pth",
        map_location="cpu"
    )
)

model.eval()


# Test accuracy
correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")