import torch

from helper_lib.model import Generator, Discriminator
from helper_lib.data_loader import get_data_loader
from helper_lib.trainer import train_gan


def main():
    torch.manual_seed(42)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using device:", device, flush=True)

    loader = get_data_loader(batch_size=64)

    generator = Generator()
    discriminator = Discriminator()

    generator, discriminator = train_gan(
        generator,
        discriminator,
        loader,
        device=device,
        epochs=5
    )

    torch.save(generator.cpu().state_dict(), "gan_generator.pth")
    torch.save(discriminator.cpu().state_dict(), "gan_discriminator.pth")

    print("GAN models saved successfully!")


if __name__ == "__main__":
    main()
