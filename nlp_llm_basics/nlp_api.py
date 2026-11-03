"""
Sentiment Analysis API

A complete NLP application that processes product reviews:
- classifies sentiment (negative / neutral / positive) with a fine-tuned DistilBERT
- extracts brands with NER
- detects the main topics with zero-shot classification
- returns structured JSON through a FastAPI REST endpoint

Install: pip install transformers datasets torch accelerate pandas scikit-learn fastapi uvicorn

Train and test:  python nlp_llm_basics/nlp_api.py
Serve the API:   uvicorn nlp_llm_basics.nlp_api:app --reload   (from the repository root)
Test it:         curl -X POST http://localhost:8000/analyse -H "Content-Type: application/json" -d '{"text": "Great battery life!"}'
"""

import os

import pandas as pd
from transformers import pipeline

MODEL_PATH = "models/sentiment_model"
label_names = {0: "negative", 1: "neutral", 2: "positive"}


# Step 1 — Data preparation
def load_reviews():
    # Sample review dataset (in practice, use a real CSV)
    reviews = [
        {"text": "Amazing battery life and gorgeous display!", "label": 2},
        {"text": "Stopped working after 3 days. Complete waste of money.", "label": 0},
        {"text": "Decent product. Does what it says.", "label": 1},
        {"text": "Best laptop I have ever owned. Highly recommend.", "label": 2},
        {"text": "Poor build quality. Screen cracked easily.", "label": 0},
        {"text": "Average performance. Nothing exceptional.", "label": 1},
        {"text": "Fast shipping, product is exactly as described.", "label": 2},
        {"text": "Very disappointed. Does not match the description.", "label": 0},
    ]

    df = pd.DataFrame(reviews)
    print(df['label'].value_counts())
    return df


# Step 2 — Fine-tune DistilBERT
def train_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset

    df = load_reviews()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenise(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenise, batched=True)
    # .values() returns the train and test splits in that order.
    train_ds, test_ds = dataset.train_test_split(test_size=0.25, seed=42).values()

    # id2label is saved with the model, so the pipeline returns "positive" instead of "LABEL_2".
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=3,
        id2label=label_names,
        label2id={name: i for i, name in label_names.items()},
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=MODEL_PATH,
            num_train_epochs=4,
            per_device_train_batch_size=8,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    trainer.save_model(MODEL_PATH)
    # save_model() does not save the tokenizer, but the pipeline in Step 3 loads it from here.
    tokenizer.save_pretrained(MODEL_PATH)


# Step 3 — Build the analysis engine
class ReviewEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.sentiment = pipeline(
            "text-classification",
            model=model_path,
            tokenizer=model_path,
        )
        self.ner = pipeline("ner", aggregation_strategy="simple")
        self.zero_shot = pipeline("zero-shot-classification",
                                  model="facebook/bart-large-mnli")
        self.topics = ["battery life", "display", "performance",
                       "price", "build quality", "customer service"]

    def analyse(self, text):
        # Sentiment
        sent = self.sentiment(text)[0]

        # Entity extraction
        entities = self.ner(text)
        brands   = [e['word'] for e in entities if e['entity_group'] in ('ORG', 'MISC')]

        # Topic detection
        # multi_label=True scores each topic independently, so a review can mention several.
        topic_result = self.zero_shot(text, candidate_labels=self.topics, multi_label=True)
        top_topics   = [label for label, score in
                       zip(topic_result['labels'], topic_result['scores']) if score > 0.4]

        return {
            "sentiment":  sent['label'],
            "confidence": round(sent['score'], 3),
            # set() removes duplicates when a brand is mentioned twice.
            "brands":     list(set(brands)),
            "topics":     top_topics[:3],
            "text":       text,
        }


# Train only when there is no saved model, so starting the API does not retrain every time.
if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    train_model()

# The engine loads all the models once, when the file is imported.
engine = ReviewEngine()


# Step 4 — FastAPI endpoint
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Sentiment Analysis API")


# Pydantic models validate the request body and describe the JSON response.
class ReviewRequest(BaseModel):
    text: str


class ReviewResponse(BaseModel):
    sentiment:  str
    confidence: float
    brands:     list
    topics:     list
    text:       str


# @app.post registers the function as the handler for POST /analyse.
@app.post("/analyse", response_model=ReviewResponse)
def analyse_review(req: ReviewRequest):
    return engine.analyse(req.text)


# A health check lets a server or load balancer confirm the API is running.
@app.get("/health")
def health():
    return {"status": "ok"}


# Step 5 — Batch processing & evaluation
# __name__ == "__main__" is only true when the file is run directly, not when uvicorn imports it.
if __name__ == "__main__":
    test_reviews = [
        "The Apple MacBook Pro has incredible performance but the price is steep.",
        "Samsung display is outstanding, battery life could be better.",
        "Poor customer service from Sony. Product broke in a week.",
    ]

    for review in test_reviews:
        result = engine.analyse(review)
        print(f"Text:      {result['text'][:60]}...")
        print(f"Sentiment: {result['sentiment']} ({result['confidence']:.0%})")
        print(f"Brands:    {result['brands']}")
        print(f"Topics:    {result['topics']}")
        print()

    # With only 8 training reviews (2 for testing), the sentiment model is a demo, not a reliable classifier.
