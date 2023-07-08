from datasets import load_dataset
from transformers import pipeline, TrainingArguments
from treasure_trove.core import filter_dataset, label_dataset, train_labeler


ds = load_dataset("CarperAI/textbooks_A2YN_labeled")["train"]
batch_size = 32
training_args = TrainingArguments(
    output_dir="./code_edu",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    seed=42,
    push_to_hub=True,
    hub_model_id="CarperAI/code_edu_classifier_py",
    hub_private_repo=True,
)
base_model_name = "bigcode/starencoder"
model, tokenizer = train_labeler(
    ds,
    "content",
    base_model_name,
    n_labels=2,
    training_args=training_args,
    num_workers=4,
    max_length=512,
    push_to_hub=True,
)