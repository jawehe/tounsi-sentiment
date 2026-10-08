from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Tounsi Sentiment API", version="1.0")

model = joblib.load("models/sentiment_model_char.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer_char.pkl")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
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

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    cleaned = clean_text(input.text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    proba = model.predict_proba(vectorized)[0]
    confidence = float(max(proba))

    if confidence < 0.6:
        sentiment = "Incertain"
    else:
        sentiment = "Positif" if prediction == 1 else "Négatif"

    return PredictionOutput(text=input.text, sentiment=sentiment, confidence=confidence)