"""
Fine-Tuning Language Models on Your Own Data

Pretrained models know a lot about language but nothing specific about your domain.
Fine-tuning adapts them to your task using a small amount of labelled data.
Covers fine-tuning DistilBERT, making predictions and LoRA.

Install: pip install transformers datasets torch accelerate peft
"""

# When to fine-tune vs prompt:
#   Prompt engineering  Generic tasks, no training data   No compute, fast          Limited control
#   Few-shot prompting  < 50 examples                     Quick                     Inconsistent
#   Fine-tuning         Domain-specific, need precision   High accuracy             Needs labelled data + GPU


# 1. Fine-tuning BERT for text classification: pretrained encoder + new classification head.
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np

# Sample sentiment data
texts = [
    "This product is absolutely amazing!",
    "Terrible quality, do not buy.",
    "Works as expected, nothing special.",
    "Best purchase I have made all year.",
    "Stopped working after one week.",
    "Decent value for the price.",
]
labels = [1, 0, 1, 1, 0, 1]  # 1=positive, 0=negative

# Load BERT tokeniser
# AutoTokenizer picks the right tokeniser class for whichever model name is given.
model_name = "distilbert-base-uncased"   # lighter version of BERT
tokenizer  = AutoTokenizer.from_pretrained(model_name)


# Tokenise
def tokenise(batch):
    return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)


dataset = Dataset.from_dict({'text': texts, 'label': labels})
dataset = dataset.map(tokenise, batched=True)
dataset = dataset.train_test_split(test_size=0.2, seed=42)

# Load model with classification head
# The head's weights start random, so transformers warns that they still need training.
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2)

# Training configuration
# evaluation_strategy was renamed eval_strategy in newer transformers versions.
args = TrainingArguments(
    output_dir='models/sentiment-results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    logging_steps=10,
)


# Trainer calls compute_metrics() after each evaluation with the logits and true labels.
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    compute_metrics=compute_metrics,
)
trainer.train()


# 2. Making predictions.
import torch


def classify(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    # Move the inputs to the same device (CPU or GPU) as the model.
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0]
    # item() turns a one-value tensor into a plain Python number.
    label = probs.argmax().item()
    return f"{'Positive' if label==1 else 'Negative'} ({probs[label]:.1%} confidence)"


print(classify("I absolutely love this product!"))
print(classify("Complete waste of money."))


# 3. LoRA (Low-Rank Adaptation): fine-tune large models cheaply.
# Instead of updating every weight, LoRA inserts small trainable matrices into existing layers.
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=16,           # rank — smaller = fewer params
    lora_alpha=32,  # scaling factor
    target_modules=["q_lin", "v_lin"],  # which layers to adapt
    lora_dropout=0.1,
)

# Wrap any HuggingFace model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: ~300K out of 67M (0.45%) !
# The LoRA matrices alone are ~300K. peft also trains the classification head,
# so the printed number can be higher.


# Practical tips:
# - Start with DistilBERT: 40% smaller, 60% faster, 97% of BERT performance.
# - Freeze early layers: only fine-tune the last 2-3 transformer blocks for small datasets.
# - Learning rate: use 2e-5 to 5e-5 (much lower than training from scratch).
# - Batch size: 8-32 depending on GPU memory; use gradient accumulation for larger effective batches.
# - Epochs: typically 3-5 is enough; more risks overfitting.
