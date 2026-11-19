import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        frequencies = torch.exp(
            -math.log(10000)
            * torch.arange(
                half_dim, device=t.device, dtype=torch.float32
            )
            / half_dim
        )
        angles = t.float()[:, None] * frequencies[None, :]
        return torch.cat(
            [torch.sin(angles), torch.cos(angles)], dim=-1
        )


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim=64):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1
        )
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1
        )
        self.time_layer = nn.Linear(time_dim, out_channels)

        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

    def forward(self, x, t):
        h = F.silu(self.norm1(self.conv1(x)))

        time_embedding = self.time_layer(t)
        h = h + time_embedding[:, :, None, None]

        h = F.silu(self.norm2(self.conv2(h)))
        return h


class SmallUNet(nn.Module):
    def __init__(self, timesteps=100):
        super().__init__()

        self.timesteps = timesteps

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(64),
            nn.Linear(64, 64),
            nn.SiLU()
        )

        self.down1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.middle = ConvBlock(64, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )
        self.dec1 = ConvBlock(128, 64)

        self.up2 = nn.ConvTranspose2d(
            64, 32, kernel_size=2, stride=2
        )
        self.dec2 = ConvBlock(64, 32)

        self.output = nn.Conv2d(32, 3, kernel_size=1)

        # Cosine noise schedule
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        alpha_bar = torch.cos(
            ((steps / timesteps + 0.008) / 1.008)
            * math.pi / 2
        ) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]

        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(0.0001, 0.999).float()
        alphas = 1.0 - betas

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer(
            "alpha_bars", torch.cumprod(alphas, dim=0)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)

        x1 = self.down1(x, t_emb)
        x2 = self.down2(self.pool1(x1), t_emb)

        middle = self.middle(self.pool2(x2), t_emb)

        u1 = self.up1(middle)
        u1 = self.dec1(torch.cat([u1, x2], dim=1), t_emb)

        u2 = self.up2(u1)
        u2 = self.dec2(torch.cat([u2, x1], dim=1), t_emb)

        return self.output(u2)

    def training_loss(self, x):
        batch_size = x.shape[0]

        t = torch.randint(
            0, self.timesteps,
            (batch_size,),
            device=x.device
        )

        noise = torch.randn_like(x)

        alpha_bar = self.alpha_bars[t].view(-1, 1, 1, 1)

        noisy_x = (
            torch.sqrt(alpha_bar) * x
            + torch.sqrt(1 - alpha_bar) * noise
        )

        predicted_noise = self(noisy_x, t)

        return F.mse_loss(predicted_noise, noise)


def sample_unet(model, batch_size=4, device="cpu"):
    model.eval()

    x = torch.randn(batch_size, 3, 32, 32, device=device)

    with torch.no_grad():
        for step in reversed(range(model.timesteps)):
            t = torch.full(
                (batch_size,),
                step,
                dtype=torch.long,
                device=device
            )

            predicted_noise = model(x, t)

            alpha = model.alphas[step]
            alpha_bar = model.alpha_bars[step]
            beta = model.betas[step]

            x = (
                x - beta / torch.sqrt(1 - alpha_bar)
                * predicted_noise
            ) / torch.sqrt(alpha)

            if step > 0:
                alpha_bar_prev = model.alpha_bars[step - 1]

                posterior_variance = (
                    beta
                    * (1 - alpha_bar_prev)
                    / (1 - alpha_bar)
                )

                x = x + torch.sqrt(
                    posterior_variance.clamp(min=1e-20)
                ) * torch.randn_like(x)

    return ((x.clamp(-1, 1) + 1) / 2).detach()
