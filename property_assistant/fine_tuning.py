"""
Fine-Tuning Language Models

Fine-tunes DistilBERT to predict property suitability from text built from properties.csv,
then compares full fine-tuning with LoRA and with freezing early layers.

Install: pip install transformers datasets torch accelerate peft pandas
"""

# 1. Load the properties and turn each row into labelled text.
import os

import numpy as np
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "properties.csv")
properties = pd.read_csv(CSV_PATH)


def describe_listing(row) -> str:
    furnished = "furnished" if row["furnished"] == "Yes" else "unfurnished"
    return (
        f"User searching in {row['user_location']}. "
        f"{row['bedrooms']} bedroom, {row['bathrooms']} bathroom "
        f"{furnished} {row['property_type']} in {row['property_location']}. "
        f"Price: KES {row['price_kes']:,}."
    )


texts = properties.apply(describe_listing, axis=1).tolist()
labels = (properties["suitable"] == "Yes").astype(int).tolist()  # 1=suitable, 0=not suitable

print(f"Dataset: {len(texts)} labelled properties")
print("Example:", texts[0], "→", labels[0])


# 2. Tokenise and split the data.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset

model_name = "distilbert-base-uncased"   # lighter version of BERT
tokenizer  = AutoTokenizer.from_pretrained(model_name)


def tokenise(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=64)


dataset = Dataset.from_dict({"text": texts, "label": labels})
dataset = dataset.map(tokenise, batched=True)
dataset = dataset.train_test_split(test_size=0.3, seed=42)

print(f"Training: {len(dataset['train'])} | Validation: {len(dataset['test'])}")

# Baseline: always predicting the most common label in the training set.
train_labels = np.array(dataset["train"]["label"])
test_labels = np.array(dataset["test"]["label"])
majority_label = np.bincount(train_labels).argmax()
baseline_accuracy = (test_labels == majority_label).mean()
print(f"Baseline accuracy (always predict {majority_label}): {baseline_accuracy:.1%}")


# 3. A reusable training function for the three approaches.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}


def count_trainable(model):
    # numel() counts the numbers in a weight tensor; requires_grad means it will be trained.
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_and_evaluate(model, learning_rate, output_dir):
    # save_strategy="no" skips checkpoints, because each one is hundreds of MB.
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=5,
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return trainer.evaluate()["eval_accuracy"]


results = {}


# 4. Full fine-tuning: every DistilBERT weight is updated.
full_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
full_trainable = count_trainable(full_model)

# 2e-5 to 5e-5 is the usual fine-tuning learning rate range.
results["Full fine-tuning"] = (
    train_and_evaluate(full_model, 5e-5, "models/property-full"),
    full_trainable,
)


# 5. Making predictions with the fine-tuned model.
import torch


def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    inputs = {key: value.to(full_model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = full_model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    label = probs.argmax().item()
    return f"{'Suitable' if label == 1 else 'Not suitable'} ({probs[label]:.1%} confidence)"


new_properties = [
    "User searching in Kilimani. 2 bedroom, 2 bathroom furnished Apartment in Kilimani. Price: KES 80,000.",
    "User searching in Ruiru. 4 bedroom, 3 bathroom unfurnished Maisonette in Westlands. Price: KES 200,000.",
]

print("\n=== Predictions on New Properties ===")
for text in new_properties:
    print(f"{text}\n  → {classify(text)}")


# 6. LoRA: train small adapter matrices instead of the whole model.
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,           # rank — smaller = fewer params
    lora_alpha=32,  # scaling factor
    target_modules=["q_lin", "v_lin"],  # DistilBERT's query and value layers
    lora_dropout=0.1,
)

lora_model = get_peft_model(
    AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2),
    lora_config,
)
print("\n=== LoRA ===")
lora_model.print_trainable_parameters()

# LoRA usually needs a higher learning rate, because far fewer weights are learning.
results["LoRA"] = (
    train_and_evaluate(lora_model, 5e-4, "models/property-lora"),
    count_trainable(lora_model),
)

# Only the small adapter needs saving, not a full copy of DistilBERT.
ADAPTER_DIR = "models/property-lora-adapter"
lora_model.save_pretrained(ADAPTER_DIR)

adapter_mb = sum(
    os.path.getsize(os.path.join(ADAPTER_DIR, name)) for name in os.listdir(ADAPTER_DIR)
) / 1e6
full_mb = sum(p.numel() for p in full_model.parameters()) * 4 / 1e6  # 4 bytes per float32
print(f"Saved LoRA adapter: {adapter_mb:.1f} MB vs full model: about {full_mb:.0f} MB")


# 7. Freeze early layers: only fine-tune the last 2 transformer blocks and the head.
frozen_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

for param in frozen_model.distilbert.embeddings.parameters():
    param.requires_grad = False

# DistilBERT has 6 transformer blocks; [:-2] selects all except the last 2.
for block in frozen_model.distilbert.transformer.layer[:-2]:
    for param in block.parameters():
        param.requires_grad = False

results["Last 2 blocks only"] = (
    train_and_evaluate(frozen_model, 5e-5, "models/property-frozen"),
    count_trainable(frozen_model),
)


# 8. Compare the approaches.
total_params = sum(p.numel() for p in full_model.parameters())

print("\n=== Fine-Tuning Comparison ===")
print(f"Baseline (no model): {baseline_accuracy:.1%}")
for name, (accuracy, trainable) in results.items():
    print(
        f"{name:<20} accuracy: {accuracy:.1%} | "
        f"trainable params: {trainable:,} ({trainable / total_params:.2%})"
    )

print("\nWith only 50 properties (15 for validation), these accuracies change a lot between splits.")
