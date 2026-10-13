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
cnn_model.load_state_dict(torch.load("cnn_model.pth", map_location="cpu"))
cnn_model.eval()

# Images need to be 32 x 32 for CIFAR-10
image_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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