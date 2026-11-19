import torch
from torchvision.utils import save_image

from src.sps_genai.diffusion_unet import SmallUNet, sample_unet

# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load trained model
model = SmallUNet(timesteps=100).to(device)

model.load_state_dict(
    torch.load(
        "diffusion_unet_model_v2.pth",
        map_location=device,
        weights_only=True
    )
)

model.eval()

# Generate images
images = sample_unet(
    model,
    batch_size=4,
    device=device
)

# Save images
save_image(
    images.cpu(),
    "diffusion_unet_samples_v2.png",
    nrow=2
)

print("Generated images:", images.shape)
print("Saved to diffusion_unet_samples_v2.png")
