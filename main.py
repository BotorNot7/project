from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from collections import Counter
import torch
import torch.nn.functional as F
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="Bot or Not API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = None
model = None

id2label = {
    0: "HUMAN",
    1: "AI"
}

class PredictRequest(BaseModel):
    text: str

def load_model():
    global tokenizer, model

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(BASE_DIR, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(BASE_DIR, local_files_only=True)
        model.eval()

    return tokenizer, model

def analyze_text(text):
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text.strip())
        if sentence.strip()
    ]

    words = re.findall(r"\b[\w']+\b", text.lower())
    unique_words = set(words)

    punctuation_count = len(re.findall(r"[^\w\s]", text))
    avg_sentence_length = round(len(words) / len(sentences), 2) if sentences else 0

    return {
        "sentence_count": len(sentences),
        "word_count": len(words),
        "unique_word_count": len(unique_words),
        "punctuation_count": punctuation_count,
        "average_sentence_length": avg_sentence_length
    }

def make_explanation(prediction, confidence, ai_probability, human_probability):
    confidence_percent = round(confidence * 100, 2)

    if prediction == "AI":
        return f"The model classified this text as AI-written with {confidence_percent}% confidence. The AI probability was higher than the human probability."

    return f"The model classified this text as human-written with {confidence_percent}% confidence. The human probability was higher than the AI probability."

@app.get("/")
def root():
    return {
        "message": "Bot or Not API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "ok"
    }

@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()

    if text == "":
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    loaded_tokenizer, loaded_model = load_model()

    inputs = loaded_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)[0]

    human_probability = float(probabilities[0])
    ai_probability = float(probabilities[1])

    predicted_id = int(torch.argmax(probabilities).item())
    prediction = id2label[predicted_id]

    confidence = max(human_probability, ai_probability)

    input_ids = inputs["input_ids"][0].tolist()
    raw_tokens = loaded_tokenizer.convert_ids_to_tokens(input_ids)

    special_tokens = set(loaded_tokenizer.all_special_tokens)

    tokens = [
        token
        for token in raw_tokens
        if token not in special_tokens
    ]

    token_counts = Counter(tokens)

    top_tokens = [
        {
            "token": token,
            "count": count
        }
        for token, count in token_counts.most_common(15)
    ]

    linguistic_analysis = analyze_text(text)

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100, 2),
        "ai_probability": round(ai_probability, 4),
        "ai_probability_percent": round(ai_probability * 100, 2),
        "human_probability": round(human_probability, 4),
        "human_probability_percent": round(human_probability * 100, 2),
        "explanation": make_explanation(
            prediction,
            confidence,
            ai_probability,
            human_probability
        ),
        "token_count": len(tokens),
        "tokens": tokens,
        "top_tokens": top_tokens,
        "linguistic_analysis": linguistic_analysis,
        "sentence_count": linguistic_analysis["sentence_count"],
        "word_count": linguistic_analysis["word_count"],
        "unique_word_count": linguistic_analysis["unique_word_count"]
    }
