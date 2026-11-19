import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.sps_genai.diffusion_unet import SmallUNet

# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device, flush=True)

# Load CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,) * 3, (0.5,) * 3)
])

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transform
)

dataset = Subset(dataset, range(10000))

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

# Initialize model
model = SmallUNet(timesteps=100).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0003
)

epochs = 3

# Training
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)

        optimizer.zero_grad()

        loss = model.training_loss(images)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {avg_loss:.4f}",
        flush=True
    )

# Save model
torch.save(
    model.cpu().state_dict(),
    "diffusion_unet_model.pth"
)

print("Small UNet model saved successfully!")
