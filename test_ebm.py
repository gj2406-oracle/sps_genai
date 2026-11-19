import torch
from torchvision.utils import save_image

from src.sps_genai.ebm import EnergyModel, sample_ebm

# Load trained model
model = EnergyModel()

model.load_state_dict(
    torch.load(
        "ebm_model.pth",
        map_location="cpu",
        weights_only=True
    )
)

model.eval()

# Generate images
images = sample_ebm(
    model,
    batch_size=4,
    steps=30,
    step_size=0.1
)

# Save images
save_image(images, "ebm_samples.png", nrow=2)

print("Generated images:", images.shape)
print("Saved to ebm_samples.png")
