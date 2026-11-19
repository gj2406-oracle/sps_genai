import torch
import torch.nn as nn


class EnergyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def sample_ebm(model, batch_size=1, steps=30, step_size=0.1,
               image_size=32, device="cpu"):
    model.eval()

    x = torch.rand(
        batch_size, 3, image_size, image_size,
        device=device
    )

    for _ in range(steps):
        x = x.detach().requires_grad_(True)

        energy = model(x).sum()

        grad = torch.autograd.grad(energy, x)[0]

        with torch.no_grad():
            x = x - step_size * grad
            x = x + 0.01 * torch.randn_like(x)
            x = x.clamp(0, 1)

    return x.detach()
