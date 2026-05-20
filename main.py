import os
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR))

MAX_LENGTH = 512
MAX_TOKENS_TO_EXPLAIN = 80
TOP_K_TOKENS = 15

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

id2label = {
    int(key): value
    for key, value in model.config.id2label.items()
}

app = FastAPI(
    title="AI vs Human Text Detector",
    description="DistilBERT backend for React morphing sphere website",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    text: str


class TopToken(BaseModel):
    token: str
    influence: float


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

    word_frequency = {}

    for word in words:
        word_frequency[word] = word_frequency.get(word, 0) + 1

    most_common_words = sorted(
        word_frequency.items(),
        key=lambda item: item[1],
        reverse=True
    )[:15]

    return {
        "sentence_count": sentence_count,
        "word_count": word_count,
        "unique_word_count": unique_word_count,
        "average_sentence_length": average_sentence_length,
        "lexical_diversity": lexical_diversity,
        "most_common_words": [
            {
                "word": word,
                "count": count
            }
            for word, count in most_common_words
        ]
    }


def get_model_inputs(text: str) -> Dict[str, torch.Tensor]:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


def predict_probabilities(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)[0]

    return probabilities


def explain_with_masking(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    predicted_id: int,
    original_confidence: float
) -> List[Dict[str, Any]]:
    special_token_ids = set(tokenizer.all_special_ids)

    if tokenizer.mask_token_id is not None:
        replacement_token_id = tokenizer.mask_token_id
    else:
        replacement_token_id = tokenizer.unk_token_id

    token_ids = input_ids[0].detach().cpu().tolist()

    positions_to_check = []

    for position, token_id in enumerate(token_ids):
        if token_id not in special_token_ids:
            positions_to_check.append(position)

    positions_to_check = positions_to_check[:MAX_TOKENS_TO_EXPLAIN]

    explanations = []

    batch_size = 16

    for start in range(0, len(positions_to_check), batch_size):
        batch_positions = positions_to_check[start:start + batch_size]

        masked_input_ids = input_ids.repeat(len(batch_positions), 1)
        masked_attention_mask = attention_mask.repeat(len(batch_positions), 1)

        for row_index, position in enumerate(batch_positions):
            masked_input_ids[row_index, position] = replacement_token_id

        with torch.no_grad():
            masked_outputs = model(
                input_ids=masked_input_ids,
                attention_mask=masked_attention_mask
            )

        masked_probabilities = F.softmax(masked_outputs.logits, dim=-1)

        for row_index, position in enumerate(batch_positions):
            masked_confidence = float(masked_probabilities[row_index, predicted_id])
            influence = original_confidence - masked_confidence

            token_id = int(input_ids[0, position].detach().cpu().item())
            token = tokenizer.convert_ids_to_tokens(token_id)

            explanations.append({
                "token": clean_token(token),
                "influence": influence
            })

    explanations = sorted(
        explanations,
        key=lambda item: item["influence"],
        reverse=True
    )

    return explanations[:TOP_K_TOKENS]


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


@app.post("/predict")
def predict(request: PredictRequest):
    text = request.text.strip()

    if text == "":
        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )

    inputs = get_model_inputs(text)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    probabilities = predict_probabilities(inputs)

    human_probability = float(probabilities[0])
    ai_probability = float(probabilities[1])

    predicted_id = int(torch.argmax(probabilities).item())
    prediction = id2label.get(predicted_id, str(predicted_id))
    confidence = float(probabilities[predicted_id])

    token_ids = input_ids[0].detach().cpu().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    cleaned_tokens = [
        clean_token(token)
        for token in tokens
    ]

    non_special_tokens = [
        token
        for token, token_id in zip(cleaned_tokens, token_ids)
        if token_id not in tokenizer.all_special_ids
    ]

    top_tokens = explain_with_masking(
        input_ids=input_ids,
        attention_mask=attention_mask,
        predicted_id=predicted_id,
        original_confidence=confidence
    )

    linguistic_analysis = analyze_text_linguistically(text)

    explanation = (
        f"The model predicted {prediction} because that label had the highest "
        f"probability. The masking test checks which tokens changed the model's "
        f"confidence the most when hidden."
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
        "linguistic_analysis": linguistic_analysis
    }