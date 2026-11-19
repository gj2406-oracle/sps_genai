import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self, timesteps=100):
        super().__init__()

        self.timesteps = timesteps

        self.network = nn.Sequential(
            nn.Conv2d(4, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 3, 3, padding=1)
        )

        betas = torch.linspace(0.0001, 0.02, timesteps)
        alphas = 1.0 - betas

        self.register_buffer(
            "alpha_bars",
            torch.cumprod(alphas, dim=0)
        )
        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)

    def forward(self, x, t):
        time_channel = (
            t.float().view(-1, 1, 1, 1)
            / self.timesteps
        )
        time_channel = time_channel.expand(
            -1, 1, x.shape[2], x.shape[3]
        )

        x_input = torch.cat([x, time_channel], dim=1)
        return self.network(x_input)

    def training_loss(self, x):
        batch_size = x.shape[0]
        t = torch.randint(
            0, self.timesteps,
            (batch_size,),
            device=x.device
        )

        noise = torch.randn_like(x)

        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        noisy_images = (
            torch.sqrt(alpha_bar) * x
            + torch.sqrt(1 - alpha_bar) * noise
        )

        predicted_noise = self(noisy_images, t)

        return F.mse_loss(predicted_noise, noise)


def sample_diffusion(model, batch_size=4, device="cpu"):
    model.eval()

    x = torch.randn(batch_size, 3, 32, 32, device=device)

    with torch.no_grad():
        for step in reversed(range(model.timesteps)):
            t = torch.full(
                (batch_size,),
                step,
                device=device,
                dtype=torch.long
            )

            predicted_noise = model(x, t)

            alpha = model.alphas[step]
            alpha_bar = model.alpha_bars[step]
            beta = model.betas[step]

            x = (
                x - (beta / torch.sqrt(1 - alpha_bar))
                * predicted_noise
            ) / torch.sqrt(alpha)

            if step > 0:
                x = x + torch.sqrt(beta) * torch.randn_like(x)

    return ((x.clamp(-1, 1) + 1) / 2).detach()
