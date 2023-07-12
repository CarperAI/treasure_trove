from datasets import load_dataset
from transformers import pipeline, TrainingArguments
import evaluate
import numpy as np
import wandb

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)


dataset = load_dataset("roborovski/phi-1")["train"]
batch_size = 32
num_workers = 4
max_length = 512
push_to_hub = True
n_labels = 3
text_column = "content"

id2label = {0: "HIGH_QUALITY", 1: "MEDIUM_QUALITY", 2: "LOW_QUALITY"}
label2id = {"HIGH_QUALITY": 0, "MEDIUM_QUALITY": 1, "LOW_QUALITY": 2}

base_model_name = "bigcode/starencoder"
tokenizer = AutoTokenizer.from_pretrained(base_model_name, max_length=max_length)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    base_model_name, num_labels=n_labels, max_length=max_length, id2label=id2label, label2id=label2id
)

dataset = dataset.map(
    lambda x: tokenizer(
        x[text_column], padding="max_length", truncation=True, max_length=max_length
    ),
    batched=True,
    num_proc=num_workers,
)

dataset = dataset.train_test_split(test_size=0.1, seed=42)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

wandb.login()

wandb.init(project="phi-2-classifier")

training_args = TrainingArguments(
    output_dir="checkpoints",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42,
    push_to_hub=True,
    hub_model_id="roborovski/phi-2-classifier",
    hub_private_repo=True,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

if push_to_hub:
    trainer.push_to_hub()
