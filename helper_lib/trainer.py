import torch
import torch.nn as nn
import torch.optim as optim


def train_gan(generator, discriminator, data_loader, device="cpu", epochs=5):
    criterion = nn.BCELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    generator.to(device)
    discriminator.to(device)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        for real_images, _ in data_loader:
            real_images = real_images.to(device)
            batch_size = real_images.size(0)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Train Discriminator
            optimizer_d.zero_grad()

            real_output = discriminator(real_images)
            loss_real = criterion(real_output, real_labels)

            noise = torch.randn(batch_size, 100, device=device)
            fake_images = generator(noise)

            fake_output = discriminator(fake_images.detach())
            loss_fake = criterion(fake_output, fake_labels)

            loss_d = loss_real + loss_fake
            loss_d.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()

            fake_output = discriminator(fake_images)
            loss_g = criterion(fake_output, real_labels)

            loss_g.backward()
            optimizer_g.step()

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss D: {loss_d.item():.4f}, "
            f"Loss G: {loss_g.item():.4f}",
            flush=True
        )

    return generator, discriminator
