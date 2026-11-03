"""
Practical NLP — Classification, NER & Summarisation

Applies NLP to the tasks that appear most often in real projects, using
HuggingFace pipelines: classification, sentiment, NER, summarisation,
question answering, text generation, and combining them into one system.

Install: pip install transformers torch
The first run downloads each model (some are over 1 GB).
"""

# 1. Text classification: assign a category to a piece of text.
# pipeline() loads a ready-to-use model and its tokeniser for a named task.
from transformers import pipeline

# Zero-shot classification — no fine-tuning needed
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")

text = "The ECB raised interest rates by 25 basis points amid ongoing inflation concerns."
labels = ["finance", "sports", "technology", "politics", "health"]

# Results come back sorted, so index 0 is the most likely label.
result = classifier(text, candidate_labels=labels)
print("Label:", result['labels'][0])
print("Score:", result['scores'][0])
# For higher accuracy, fine-tune a classifier on your own labelled data (previous lesson).


# 2. Sentiment analysis.
# With no model named, the pipeline uses a default English model (POSITIVE or NEGATIVE only),
# so a neutral text is still forced into one of the two.
sentiment = pipeline("sentiment-analysis")

texts = [
    "This restaurant has the best pasta I have ever eaten!",
    "The service was slow and the food was cold.",
    "It was okay, nothing to write home about.",
]

print()
for text in texts:
    result = sentiment(text)[0]
    print(f"{result['label']:8s} ({result['score']:.0%})  {text}")


# 3. Named Entity Recognition (NER): find people, organisations and locations.
# aggregation_strategy="simple" joins word pieces like "Mu" + "##sk" into whole entities.
ner = pipeline("ner", aggregation_strategy="simple")

text = "Elon Musk announced that Tesla will open a new factory in Berlin by March 2025."
entities = ner(text)

print()
for e in entities:
    print(f"{e['word']:20s}  {e['entity_group']:10s}  score={e['score']:.2f}")
# Expected:
# Elon Musk             PER         score=0.99
# Tesla                 ORG         score=0.99
# Berlin                LOC         score=0.98
# The default model only knows PER, ORG, LOC and MISC, so "March 2025" is not tagged as a DATE.


# 4. Text summarisation.
summariser = pipeline("summarization", model="facebook/bart-large-cnn")

article = """
The James Webb Space Telescope has captured unprecedented images of galaxies
forming just 400 million years after the Big Bang. Scientists at NASA say the
observations challenge existing models of galaxy formation and could require
a fundamental rethinking of how structures emerged in the early universe.
The telescope, launched in December 2021, uses infrared imaging to peer through
dust clouds and observe some of the oldest light in the universe.
"""

# max_length and min_length are measured in tokens; do_sample=False gives the same summary every time.
summary = summariser(article, max_length=60, min_length=20, do_sample=False)
print("\n" + summary[0]['summary_text'])


# 5. Question answering: find the answer inside a given context.
# The answer is copied from the context; the model does not write new text.
qa = pipeline("question-answering")

context = """
Python was created by Guido van Rossum and first released in 1991.
It emphasises code readability and supports multiple programming paradigms.
Python is widely used in data science, web development, and artificial intelligence.
"""

questions = [
    "Who created Python?",
    "When was Python first released?",
    "What is Python used for?",
]

print()
for q in questions:
    answer = qa(question=q, context=context)
    print(f"Q: {q}")
    print(f"A: {answer['answer']}  (score={answer['score']:.2f})\n")


# 6. Text generation.
generator = pipeline("text-generation", model="gpt2")

prompt = "The future of artificial intelligence is"
# do_sample=True turns on random sampling, so temperature has an effect.
output = generator(prompt, max_new_tokens=80, num_return_sequences=1,
                   temperature=0.8, do_sample=True)
print(output[0]['generated_text'])


# 7. Building an NLP pipeline for production: combine several tasks in one class.
class ReviewAnalyser:
    def __init__(self):
        self.sentiment   = pipeline("sentiment-analysis")
        self.ner         = pipeline("ner", aggregation_strategy="simple")
        self.summariser  = pipeline("summarization", model="facebook/bart-large-cnn")

    def analyse(self, review_text):
        sentiment  = self.sentiment(review_text)[0]
        entities   = self.ner(review_text)
        # MISC covers named things that are not people, organisations or locations, e.g. products.
        products   = [e['word'] for e in entities if e['entity_group'] == 'MISC']
        summary    = self.summariser(review_text, max_length=40, min_length=10)[0]['summary_text']

        return {
            "sentiment":  sentiment['label'],
            "confidence": round(sentiment['score'], 2),
            "products_mentioned": products,
            "summary":    summary,
        }


analyser = ReviewAnalyser()
review = "The new iPhone 15 Pro has an incredible camera system. Battery life is disappointing though."
print("\n", analyser.analyse(review))
