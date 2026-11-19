import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.sps_genai.diffusion import DiffusionModel


# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device, flush=True)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 2 - 1)
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

# Use 5000 images for initial training
dataset = Subset(dataset, range(20000))

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

# Initialize diffusion model
model = DiffusionModel(timesteps=100).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

epochs = 20

# Training loop
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)

        loss = model.training_loss(images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {avg_loss:.4f}",
        flush=True
    )

# Save trained model
torch.save(
    model.cpu().state_dict(),
    "diffusion_model_v2.pth"
)

print("Diffusion model saved successfully!")
