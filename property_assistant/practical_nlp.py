"""
Practical NLP — Property Assistant

Uses HuggingFace pipelines to understand user messages: intent classification,
sentiment, NER, summarisation, question answering over listings and text generation,
then combines them to turn a free-text request into matching listings from properties.csv.

Install: pip install transformers torch pandas
The first run downloads each model (some are over 1 GB).
"""

import os
import re

import pandas as pd
from transformers import pipeline

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)


def describe_listing(row) -> str:
    furnished = "furnished" if row["furnished"] == "Yes" else "unfurnished"
    return (
        f"Listing #{row['property_id']}: {row['bedrooms']} bedroom, "
        f"{row['bathrooms']} bathroom {furnished} {row['property_type']} "
        f"in {row['property_location']}. Price: KES {row['price_kes']:,}."
    )


properties["description"] = properties.apply(describe_listing, axis=1)
LOCATIONS = sorted(properties["property_location"].unique())

# Illustrative messages a Property Assistant might receive.
messages = [
    "Hi, I'm looking for a 2 bedroom apartment in Kilimani or Westlands, budget KES 100,000.",
    "Can I book a viewing for the Ruaka maisonette this Saturday?",
    "The agent never showed up for my viewing in Kasarani. Very disappointed.",
]


# 1. Intent classification with zero-shot: no labelled training data needed.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

intents = ["searching for a property", "booking a viewing", "making a complaint", "asking about price"]

print("=== Intent ===")
for message in messages:
    result = classifier(message, candidate_labels=intents)
    print(f"{result['labels'][0]:<25} ({result['scores'][0]:.0%})  {message}")


# 2. Sentiment: spot unhappy users so they can be followed up first.
sentiment = pipeline("sentiment-analysis")

print("\n=== Sentiment ===")
for message in messages:
    result = sentiment(message)[0]
    print(f"{result['label']:8s} ({result['score']:.0%})  {message}")


# 3. NER: find locations in a message.
ner = pipeline("ner", aggregation_strategy="simple")


# An English NER model may miss local place names, so also match against the CSV locations.
def find_locations(text):
    ner_locations = [e["word"] for e in ner(text) if e["entity_group"] == "LOC"]
    csv_locations = [location for location in LOCATIONS if location.lower() in text.lower()]
    return ner_locations, csv_locations


print("\n=== Locations ===")
for message in messages:
    ner_locations, csv_locations = find_locations(message)
    print(f"NER: {ner_locations} | Known locations: {csv_locations}  ←  {message}")


# 4. Turn the request into exact filters, then find matching listings.
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
    return matches


request = messages[0]
_, locations = find_locations(request)
minimum_bedrooms = extract_minimum_bedrooms(request)
budget = extract_budget(request)
matches = find_listings(locations, minimum_bedrooms, budget)

print("\n=== Request → Filters → Listings ===")
print(f"Request: {request}")
print(f"Filters: locations={locations}, minimum_bedrooms={minimum_bedrooms}, budget={budget}")
print(f"{len(matches)} matching listings:")
for description in matches["description"]:
    print(f"  {description}")


# 5. Question answering over the listings in one location.
qa = pipeline("question-answering")

context = "\n".join(properties.loc[properties["property_location"] == "Kilimani", "description"])

questions = [
    "How much is the 3 bedroom listing?",
    "Which listing costs KES 85,000?",
    "How many bedrooms does the cheapest listing have?",
]

print("\n=== Questions about Kilimani Listings ===")
print(context)
for q in questions:
    answer = qa(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {answer['answer']}  (score={answer['score']:.2f})\n")
# Extractive QA copies a span from the text, so it cannot compare or calculate.
# Questions like "the cheapest" are safer answered with pandas.


# 6. Summarisation: shorten a long message for the agent.
summariser = pipeline("summarization", model="facebook/bart-large-cnn")

long_message = """
Good morning. My family and I are relocating to Nairobi next month because I have
started a new job in Westlands. We have two young children, so we need at least
three bedrooms and ideally two bathrooms. We would prefer somewhere quiet, and a
furnished place would save us a lot of trouble since we are moving from abroad.
Our budget is around KES 150,000 per month. We could also consider Kileleshwa or
Kilimani if there is nothing suitable in Westlands. Please let me know what you have.
"""

summary = summariser(long_message, max_length=50, min_length=15, do_sample=False)
print("=== Summary for the Agent ===")
print(summary[0]["summary_text"])


# 7. Text generation: draft a listing advert.
generator = pipeline("text-generation", model="gpt2")

listing = properties["description"].iloc[0]
prompt = f"{listing} This home"
output = generator(prompt, max_new_tokens=60, num_return_sequences=1,
                   temperature=0.8, do_sample=True)

print("\n=== Generated Advert Draft ===")
print(output[0]["generated_text"])
# GPT-2 is not grounded in the data, so it can invent features (a pool, a view, a different price).
# Every generated advert needs checking against the real listing.


# 8. Combine the tasks into one Property Assistant pipeline.
# The pipelines loaded above are passed in, so the models are not loaded twice.
class PropertyRequestAnalyser:
    def __init__(self, classifier, sentiment):
        self.classifier = classifier
        self.sentiment  = sentiment

    def analyse(self, message):
        intent    = self.classifier(message, candidate_labels=intents)
        sentiment = self.sentiment(message)[0]
        _, locations = find_locations(message)
        minimum_bedrooms = extract_minimum_bedrooms(message)
        budget = extract_budget(message)

        result = {
            "intent":     intent["labels"][0],
            "sentiment":  sentiment["label"],
            "confidence": round(sentiment["score"], 2),
            "locations":  locations,
            "minimum_bedrooms": minimum_bedrooms,
            "budget":     budget,
        }

        # Only search the listings when the user is actually looking for a property.
        if result["intent"] == "searching for a property":
            matches = find_listings(locations, minimum_bedrooms, budget)
            result["matching_property_ids"] = matches["property_id"].tolist()

        return result


analyser = PropertyRequestAnalyser(classifier, sentiment)

print("\n=== Property Request Analyser ===")
for message in messages:
    print(message)
    print(" ", analyser.analyse(message))
