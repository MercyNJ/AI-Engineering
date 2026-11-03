"""
Property Message Analysis API

An NLP API for the Property Assistant that:
- classifies a user message's intent (search / viewing / complaint / price) with a fine-tuned DistilBERT
- finds locations with NER plus an exact match against properties.csv
- detects topics (price, security, parking, ...) with zero-shot classification
- extracts minimum bedrooms and budget, and returns matching listings for searches
- serves everything as JSON through a FastAPI REST API

Install: pip install transformers datasets torch accelerate pandas fastapi uvicorn

Train and test:  python property_assistant/nlp_api.py
Serve the API:   uvicorn property_assistant.nlp_api:app --reload   (from the repository root)
Test it:         curl -X POST http://localhost:8000/analyse -H "Content-Type: application/json" -d '{"text": "Any 2 bedroom apartments in Kilimani under KES 90,000?"}'
"""

import os
import re

import pandas as pd
from transformers import pipeline

MODEL_PATH = "models/property_intent_model"
label_names = {0: "search", 1: "viewing", 2: "complaint", 3: "price"}

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)
LOCATIONS = sorted(properties["property_location"].unique())


# Step 1 — Data preparation: labelled user messages (illustrative).
def load_messages():
    messages = [
        {"text": "Do you have any 3 bedroom houses in Ruiru?", "label": 0},
        {"text": "Looking for a furnished apartment in Kilimani.", "label": 0},
        {"text": "Show me 2 bedroom places in Westlands under 100k.", "label": 0},
        {"text": "Any maisonettes available in Ruaka?", "label": 0},
        {"text": "Can I come and see listing #12 on Friday?", "label": 1},
        {"text": "I'd like to book a viewing for the Langata house.", "label": 1},
        {"text": "Is Saturday morning free to view the Kileleshwa apartment?", "label": 1},
        {"text": "When can I visit the South B property?", "label": 1},
        {"text": "The caretaker was rude and the flat was dirty.", "label": 2},
        {"text": "Nobody fixed the leaking tap I reported last month.", "label": 2},
        {"text": "The agent never showed up for my viewing.", "label": 2},
        {"text": "The photos were misleading, the house is much smaller.", "label": 2},
        {"text": "How much is the rent for listing #7?", "label": 3},
        {"text": "Is the price of the Westlands maisonette negotiable?", "label": 3},
        {"text": "What is the deposit for the Kasarani apartment?", "label": 3},
        {"text": "Does the rent include the service charge?", "label": 3},
    ]

    df = pd.DataFrame(messages)
    print(df["label"].map(label_names).value_counts())
    return df


# Step 2 — Fine-tune DistilBERT on message intent.
def train_model():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
    from datasets import Dataset

    df = load_messages()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenise(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(tokenise, batched=True)
    train_ds, test_ds = dataset.train_test_split(test_size=0.25, seed=42).values()

    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=len(label_names),
        id2label=label_names,
        label2id={name: i for i, name in label_names.items()},
    )

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=MODEL_PATH,
            num_train_epochs=8,
            learning_rate=5e-5,
            per_device_train_batch_size=8,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="none",
        ),
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()
    trainer.save_model(MODEL_PATH)
    # save_model() does not save the tokenizer, but the pipeline in Step 3 loads it from here.
    tokenizer.save_pretrained(MODEL_PATH)


# Exact filters, so prices and bedroom counts are never left to a model.
WORD_NUMBERS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def extract_minimum_bedrooms(text):
    # Matches "2 bedroom", "2-bedroom", "two bedroom" or "3 bed".
    match = re.search(r"\b(\d+|one|two|three|four|five)[\s-]*bed", text.lower())
    if not match:
        return None
    value = match.group(1)
    return int(value) if value.isdigit() else WORD_NUMBERS[value]


def extract_budget(text):
    text = text.lower()
    # Matches "KES 100,000" or "ksh 80000".
    match = re.search(r"(?:kes|ksh)\s*([\d,]+)", text)
    if match:
        return int(match.group(1).replace(",", ""))
    # Matches "80k".
    match = re.search(r"\b(\d+)\s*k\b", text)
    if match:
        return int(match.group(1)) * 1000
    return None


def find_listings(locations, minimum_bedrooms, budget):
    matches = properties
    if locations:
        matches = matches[matches["property_location"].isin(locations)]
    if minimum_bedrooms:
        matches = matches[matches["bedrooms"] >= minimum_bedrooms]
    if budget:
        matches = matches[matches["price_kes"] <= budget]
    return matches.sort_values("price_kes")


# Step 3 — Build the analysis engine.
class PropertyMessageEngine:
    def __init__(self, model_path=MODEL_PATH):
        self.intent = pipeline("text-classification", model=model_path, tokenizer=model_path)
        self.ner = pipeline("ner", aggregation_strategy="simple")
        self.zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.topics = ["price", "location", "security", "parking",
                       "water supply", "furnishing", "maintenance"]

    def find_locations(self, text):
        # An English NER model may miss local names like Ruaka, so also match the CSV locations.
        ner_locations = [e["word"] for e in self.ner(text) if e["entity_group"] == "LOC"]
        csv_locations = [location for location in LOCATIONS if location.lower() in text.lower()]
        # Keep only names that exist in the data; NER may find places with no listings.
        return sorted(set(csv_locations) | {loc for loc in ner_locations if loc in LOCATIONS})

    def analyse(self, text):
        intent = self.intent(text)[0]
        locations = self.find_locations(text)

        topic_result = self.zero_shot(text, candidate_labels=self.topics, multi_label=True)
        top_topics = [label for label, score in
                      zip(topic_result["labels"], topic_result["scores"]) if score > 0.4]

        minimum_bedrooms = extract_minimum_bedrooms(text)
        budget = extract_budget(text)

        # Only search messages get matching listings.
        matching_ids = []
        if intent["label"] == "search":
            matches = find_listings(locations, minimum_bedrooms, budget)
            matching_ids = [int(property_id) for property_id in matches["property_id"]]

        return {
            "intent": intent["label"],
            "confidence": round(intent["score"], 3),
            "locations": locations,
            "topics": top_topics[:3],
            "minimum_bedrooms": minimum_bedrooms,
            "budget_kes": budget,
            "matching_property_ids": matching_ids,
            "text": text,
        }


# Train only when there is no saved model, so starting the API does not retrain every time.
if not os.path.exists(os.path.join(MODEL_PATH, "config.json")):
    train_model()

engine = PropertyMessageEngine()


# Step 4 — FastAPI endpoints.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Property Message Analysis API")


class MessageRequest(BaseModel):
    text: str


class MessageResponse(BaseModel):
    intent: str
    confidence: float
    locations: list[str]
    topics: list[str]
    # int | None means the field can be a number or null when nothing was found.
    minimum_bedrooms: int | None
    budget_kes: int | None
    matching_property_ids: list[int]
    text: str


class PropertyResponse(BaseModel):
    property_id: int
    property_location: str
    property_type: str
    bedrooms: int
    bathrooms: int
    price_kes: int
    furnished: str


@app.post("/analyse", response_model=MessageResponse)
def analyse_message(req: MessageRequest):
    return engine.analyse(req.text)


# {property_id} in the path is passed to the function as an int.
@app.get("/properties/{property_id}", response_model=PropertyResponse)
def get_property(property_id: int):
    match = properties[properties["property_id"] == property_id]
    if match.empty:
        # HTTPException returns a proper 404 error response instead of crashing.
        raise HTTPException(status_code=404, detail=f"No listing with id {property_id}")

    row = match.iloc[0]
    # int() and str() convert pandas values into plain Python types for the JSON response.
    return {
        "property_id": int(row["property_id"]),
        "property_location": str(row["property_location"]),
        "property_type": str(row["property_type"]),
        "bedrooms": int(row["bedrooms"]),
        "bathrooms": int(row["bathrooms"]),
        "price_kes": int(row["price_kes"]),
        "furnished": str(row["furnished"]),
    }


@app.get("/health")
def health():
    return {"status": "ok", "properties_loaded": len(properties)}


# Step 5 — Batch processing.
if __name__ == "__main__":
    test_messages = [
        "Hi, I'm looking for a 2 bedroom apartment in Kilimani or Westlands, budget KES 100,000.",
        "Can I book a viewing for the Ruaka maisonette this Saturday?",
        "The agent never showed up and there was no water when I visited Kasarani.",
        "How much is the 3 bedroom house in Ruiru? Does it have parking?",
    ]

    for message in test_messages:
        result = engine.analyse(message)
        print(f"Text:      {result['text'][:60]}...")
        print(f"Intent:    {result['intent']} ({result['confidence']:.0%})")
        print(f"Locations: {result['locations']}")
        print(f"Topics:    {result['topics']}")
        print(f"Filters:   minimum_bedrooms={result['minimum_bedrooms']}, budget_kes={result['budget_kes']}")
        print(f"Matches:   {result['matching_property_ids']}")
        print()

    # With only 16 labelled messages (4 for testing), the intent model is a demo, not a reliable classifier.
