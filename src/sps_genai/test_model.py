import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from sps_genai.cnn import SimpleCNN


transform = transforms.Compose([
    transforms.ToTensor()
])

# Load CIFAR-10 test data
test_data = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)

# Load the trained CNN
model = SimpleCNN()
model.load_state_dict(torch.load("cnn_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")