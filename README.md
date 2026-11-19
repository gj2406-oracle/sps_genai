# FastAPI ML Project

This project uses Python, PyTorch, and FastAPI to build and test different machine learning models.

The project includes word embeddings, CNN image classification, GAN image generation, Energy-Based Models (EBM), and Diffusion Models.

## 1. Word Embedding

The `/embedding` endpoint returns a word embedding using spaCy.

Example input:

```json
{
  "word": "apple"
}
```

## 2. CNN Image Classifier

The `/predict` endpoint uses a CNN model to classify an uploaded image.

The CNN was trained on the CIFAR-10 dataset using PyTorch.

The model can predict 10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## 3. GAN Image Generation

I implemented a Generative Adversarial Network (GAN) using PyTorch.

The GAN contains two models:

- Generator: creates images from random noise.
- Discriminator: tries to distinguish real images from generated images.

The `/generate` endpoint generates an image using the trained GAN generator.

## 4. Energy-Based Model (EBM)

I implemented an Energy-Based Model using PyTorch.

The model learns an energy function for images. Lower energy values should represent more likely images.

I also tested image generation using Langevin sampling.

Main files:

- `src/sps_genai/ebm.py`
- `train_ebm.py`
- `test_ebm.py`

## 5. Diffusion Model

I implemented a diffusion model using PyTorch.

The model learns to remove noise from images. During training, random noise is added to images, and the model learns to predict that noise.

I also implemented a small U-Net model to improve noise prediction.

Main files:

- `src/sps_genai/diffusion.py`
- `src/sps_genai/diffusion_unet.py`
- `train_diffusion.py`
- `train_diffusion_unet.py`
- `train_diffusion_unet_v2.py`
- `test_diffusion.py`
- `test_diffusion_unet.py`
- `test_diffusion_unet_v2.py`

The U-Net model was trained on CIFAR-10 images using PyTorch.

I tested different noise schedules and sampling methods. The training loss decreased, but some generated images still contained noise and unclear patterns.

## 6. FastAPI Endpoints

The project includes the following API endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Check API status |
| `/embedding` | POST | Generate word embeddings |
| `/predict` | POST | Classify uploaded images |
| `/generate` | GET | Generate an image using GAN |
| `/generate/diffusion` | GET | Generate an image using Diffusion |

## 7. Run the Project

Install the project dependencies and required spaCy model in your Python environment.

Start the FastAPI server:

```bash
python -m uvicorn app.main:app --reload
```

Then open:

http://127.0.0.1:8000/docs

The Swagger UI can be used to test the API endpoints.

The trained model weights are required for the image generation endpoints.

## 8. Run with Docker

Build the Docker image:

```bash
docker build -t sps-genai .
```

Run the container:

```bash
docker run --rm -p 8000:80 sps-genai
```

Then open:

http://127.0.0.1:8000/docs

The Docker setup also requires the necessary model files and dependencies.

## 9. Results

I successfully implemented and tested the EBM and Diffusion models.

For the U-Net Diffusion model, I trained the model for 20 epochs using 20,000 CIFAR-10 images.

The training loss decreased from approximately 0.2044 to 0.0654.

The FastAPI diffusion endpoint was also tested successfully and returned a generated PNG image.

However, the generated images still contained visible noise, so the image quality could be improved with more training and further model adjustments.
