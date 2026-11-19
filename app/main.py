from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import spacy
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

from sps_genai.cnn import SimpleCNN


app = FastAPI()

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"message": "API is running"}


@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    token = nlp(request.word)

    if len(token) == 0:
        return {"error": "No word provided"}

    vector = token.vector.tolist()

    return {
        "word": request.word,
        "embedding": vector
    }


# CIFAR-10 classes
CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


# Load trained CNN
cnn_model = SimpleCNN()
cnn_model.load_state_dict(
    torch.load("cnn_model.pth", map_location="cpu")
)
cnn_model.eval()


# Resize uploaded images to 64 x 64
# to match the CNN architecture
image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGB")

    image = image_transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = cnn_model(image)
        _, predicted = torch.max(output, 1)

    predicted_class = CLASSES[predicted.item()]

    return {
        "filename": file.filename,
        "predicted_class": predicted_class
    }

# GAN image generation
from helper_lib.model import Generator
from fastapi.responses import Response
from torchvision.utils import save_image

gan_generator = Generator()
gan_generator.load_state_dict(
    torch.load(
        "gan_generator.pth",
        map_location="cpu",
        weights_only=True
    )
)
gan_generator.eval()


@app.get("/generate")
def generate_image():
    with torch.no_grad():
        noise = torch.randn(1, 100)
        fake_image = gan_generator(noise)

    fake_image = (fake_image + 1) / 2

    buffer = io.BytesIO()
    save_image(fake_image, buffer, format="PNG")

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )

# Diffusion model image generation

from src.sps_genai.diffusion_unet import SmallUNet, sample_unet

diffusion_model = SmallUNet(timesteps=100)

diffusion_model.load_state_dict(
    torch.load(
        "diffusion_unet_model_v2.pth",
        map_location="cpu",
        weights_only=True
    )
)

diffusion_model.eval()


@app.get("/generate/diffusion")
def generate_diffusion_image():

    with torch.no_grad():
        generated_image = sample_unet(
            diffusion_model,
            batch_size=1,
            device="cpu"
        )

    buffer = io.BytesIO()

    save_image(
        generated_image,
        buffer,
        format="PNG"
    )

    return Response(
        content=buffer.getvalue(),
        media_type="image/png"
    )

