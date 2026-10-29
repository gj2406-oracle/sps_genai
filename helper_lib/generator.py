import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def generate_samples(model, device="cpu", num_samples=16):
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        noise = torch.randn(num_samples, 100, device=device)
        images = model(noise)

    images = (images.cpu() + 1) / 2

    grid = make_grid(images, nrow=4, padding=2)

    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.savefig("gan_samples.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Generated images saved to gan_samples.png")
