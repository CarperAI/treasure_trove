from dataclasses import dataclass
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


@dataclass
class EncoderParams:
    batch_size = 32
    num_workers = 4
    push_to_hub = True
    n_labels = 3
    text_column = "content"
    labels = ["high quality", "medium quality", "low quality"]
    base_model_name = "bigcode/starencoder"
    id2label = {0: "HIGH_QUALITY", 1: "MEDIUM_QUALITY", 2: "LOW_QUALITY"}
    label2id = {"HIGH_QUALITY": 0, "MEDIUM_QUALITY": 1, "LOW_QUALITY": 2}
    MASK_TOKEN = "<mask>"
    SEPARATOR_TOKEN = "<sep>"
    PAD_TOKEN = "<pad>"
    CLS_TOKEN = "<cls>"
    max_input_length = 10000
    max_token_length = 1024


def train():

    dataset = load_dataset("roborovski/phi-1")["train"]


    tokenizer = AutoTokenizer.from_pretrained(
        EncoderParams.base_model_name, max_length=EncoderParams.max_token_length
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSequenceClassification.from_pretrained(
        EncoderParams.base_model_name,
        num_labels=EncoderParams.n_labels,
        max_length=EncoderParams.max_token_length,
        id2label=EncoderParams.id2label,
        label2id=EncoderParams.label2id,
    )


    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if isinstance(logits, tuple):  # Some models return tuples
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        acc = acc_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro" if len(labels) > 2 else "binary",
        )
        recall = recall_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro" if len(labels) > 2 else "binary",
        )
        f1 = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="macro" if len(labels) > 2 else "binary",
        )

        return {**acc, **precision, **recall, **f1}

    dataset = dataset.map(
        lambda x: tokenizer(
            x[EncoderParams.text_column],
            padding="max_length",
            truncation=True,
            max_length=EncoderParams.max_input_length,
        ),
        batched=True,
        num_proc=EncoderParams.num_workers,
    )

    dataset = dataset.train_test_split(test_size=0.05, seed=42)

    train_dataset = dataset["train"].shuffle(seed=42)
    eval_dataset = dataset["test"].shuffle(seed=42).select(range(200))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    acc_metric = evaluate.load("accuracy")
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    wandb.login()

    wandb.init(project="phi-2-classifier")

    training_args = TrainingArguments(
        output_dir="checkpoints",
        num_train_epochs=100,
        per_device_train_batch_size=EncoderParams.batch_size,
        per_device_eval_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="logs",
        logging_steps=50,
        eval_steps=5000,
        evaluation_strategy="steps",
        save_strategy="epoch",
        save_steps=5,
        seed=42,
        push_to_hub=True,
        hub_model_id="roborovski/phi-2-classifier",
        hub_private_repo=True,
        eval_accumulation_steps=1,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    if EncoderParams.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    train()
