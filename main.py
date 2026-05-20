import os
import re
from pathlib import Path
from typing import Dict, Any
from collections import Counter

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR))

MAX_LENGTH = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = None
model = None
id2label = {
    0: "HUMAN",
    1: "AI"
}

app = FastAPI(
    title="AI vs Human Text Detector",
    description="DistilBERT backend for React morphing sphere website",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


def load_model():
    global tokenizer, model, id2label

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            use_fast=True,
            local_files_only=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True
        )

        model.to(device)
        model.eval()

        if hasattr(model.config, "id2label"):
            id2label = {
                int(key): value
                for key, value in model.config.id2label.items()
            }

    return tokenizer, model


def clean_token(token: str) -> str:
    token = token.replace("##", "")
    token = token.strip()

    if token == "":
        return "[EMPTY]"

    return token


def analyze_text_linguistically(text: str) -> Dict[str, Any]:
    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    words = re.findall(r"\b[\w']+\b", text.lower())
    unique_words = set(words)

    word_count = len(words)
    sentence_count = len(sentences)
    unique_word_count = len(unique_words)

    average_sentence_length = 0
    lexical_diversity = 0

    if sentence_count > 0:
        average_sentence_length = word_count / sentence_count

    if word_count > 0:
        lexical_diversity = unique_word_count / word_count

    word_frequency = Counter(words)

    most_common_words = [
        {
            "word": word,
            "count": count
        }
        for word, count in word_frequency.most_common(15)
    ]

    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "average_sentence_length": average_sentence_length,
        "lexical_diversity": lexical_diversity,
        "most_common_words": most_common_words
    }


@app.get("/")
def home():
    return {
        "message": "AI vs Human detector backend is running",
        "model_dir": str(MODEL_DIR),
        "device": str(device)
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device)
    }


@app.get("/debug-files")
def debug_files():
    needed_files = [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]

    result = {}

    for file_name in needed_files:
        file_path = MODEL_DIR / file_name

        result[file_name] = {
            "exists": file_path.exists(),
            "size_mb": round(file_path.stat().st_size / 1024 / 1024, 2) if file_path.exists() else 0
        }

    return result


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()

    if text == "":
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    loaded_tokenizer, loaded_model = load_model()

    inputs = loaded_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = loaded_model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)[0]

    human_probability = float(probabilities[0])
    ai_probability = float(probabilities[1])

    predicted_id = int(torch.argmax(probabilities).item())
    prediction = id2label.get(predicted_id, str(predicted_id))
    confidence = float(probabilities[predicted_id])

    input_ids = inputs["input_ids"][0].detach().cpu().tolist()
    raw_tokens = loaded_tokenizer.convert_ids_to_tokens(input_ids)

    cleaned_tokens = [
        clean_token(token)
        for token in raw_tokens
    ]

    non_special_tokens = [
        token
        for token, token_id in zip(cleaned_tokens, input_ids)
        if token_id not in loaded_tokenizer.all_special_ids
    ]

    token_frequency = Counter(non_special_tokens)

    top_tokens = [
        {
            "token": token,
            "count": count
        }
        for token, count in token_frequency.most_common(15)
    ]

    linguistic_analysis = analyze_text_linguistically(text)

    explanation = (
        f"The model predicted {prediction} because that label had the highest probability. "
        f"This deploy-safe version returns the most repeated tokens instead of running the expensive masking test."
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "human_probability": human_probability,
        "ai_probability": ai_probability,
        "explanation": explanation,
        "token_count": len(non_special_tokens),
        "tokens": non_special_tokens,
        "top_tokens": top_tokens,
        "linguistic_analysis": linguistic_analysis,
        "sentence_count": linguistic_analysis["sentence_count"],
        "word_count": linguistic_analysis["word_count"],
        "unique_word_count": linguistic_analysis["unique_word_count"]
    }
