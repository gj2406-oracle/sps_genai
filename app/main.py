from fastapi import FastAPI
from pydantic import BaseModel
import spacy

app = FastAPI()

# Load spaCy model with word vectors
nlp = spacy.load("en_core_web_md")


class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"message": "Word Embedding API is running"}


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