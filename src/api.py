import re

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Tounsi Sentiment API", version="1.1")

model = joblib.load("models/sentiment_model_char.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer_char.pkl")

CONFIDENCE_THRESHOLD = 0.6


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


class TextInput(BaseModel):
    text: str


class PredictionOutput(BaseModel):
    text: str
    sentiment: str
    confidence: float


@app.get("/")
def root():
    return {"message": "Tounsi Sentiment API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: TextInput):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Le texte est vide")

    cleaned = clean_text(payload.text)
    if not cleaned:
        raise HTTPException(status_code=400, detail="Le texte ne contient aucun mot exploitable")

    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = float(max(proba))

    if confidence < CONFIDENCE_THRESHOLD:
        sentiment = "Incertain"
    else:
        sentiment = "Positif" if prediction == 1 else "Négatif"

    return PredictionOutput(text=payload.text, sentiment=sentiment, confidence=confidence)