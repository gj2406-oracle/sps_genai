# FastAPI ML Project

This project contains two API functions built with FastAPI.

## Word Embedding

The `/embedding` endpoint returns a word embedding using spaCy.

Example input:

{
  "word": "apple"
}

## CNN Image Classifier

The `/predict` endpoint uses a CNN model to classify an uploaded image.

The CNN was trained on the CIFAR-10 dataset using PyTorch. The model contains convolution, max pooling, and fully connected layers.

The model can predict 10 classes:

airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Run with Docker

Build the Docker image:

docker build -t sps-genai .

Run the container:

docker run --rm -p 8000:80 sps-genai

Then open:

http://127.0.0.1:8000/docs

The Swagger page can be used to test both `/embedding` and `/predict`.