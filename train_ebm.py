import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from src.sps_genai.ebm import EnergyModel, sample_ebm


device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Using device:", device, flush=True)

dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

dataset = Subset(dataset, range(1000))

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

model = EnergyModel().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)

epochs = 2

for epoch in range(epochs):
    total_loss = 0

    for images, _ in loader:
        images = images.to(device)

        # Generate negative samples with Langevin dynamics
        negative_images = sample_ebm(
            model,
            batch_size=images.size(0),
            steps=10,
            step_size=0.1,
            device=device
        )

        model.train()

        positive_energy = model(images)
        negative_energy = model(negative_images)

        # Contrastive loss with energy regularization
        loss = (
            positive_energy.mean()
            - negative_energy.mean()
            + 0.01 * (
                positive_energy.square().mean()
                + negative_energy.square().mean()
            )
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {total_loss / len(loader):.4f}",
        flush=True
    )

torch.save(
    model.cpu().state_dict(),
    "ebm_model_langevin.pth"
)

print("Langevin EBM model saved successfully!")
