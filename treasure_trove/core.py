# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/00_core.ipynb.

# %% auto 0
__all__ = ['classify', 'label_dataset', 'train_labeler', 'filter_dataset']

# %% ../nbs/00_core.ipynb 2
import evaluate
import time

import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

# %% ../nbs/00_core.ipynb 4
def classify(x, labels, llm_labeler, max_failures=5, default_label=0):
    failures = 0
    while failures < max_failures:
        try:
            return labels.index(llm_labeler(x)[0])
        except Exception as e:
            failures += 1
            print(e)
            time.sleep(1)
            pass
    if failures == max_failures:
        return default_label

# %% ../nbs/00_core.ipynb 5
def label_dataset(
    dataset, text_column, labeler_model, labels, sample=0.1, num_workers=4
):
    """
    Filters a dataset using a labeler model.

    Args:
        dataset (datasets.Dataset): Dataset to filter
        text_column (str): Name of the column containing the text to classify
        labeler_model (Any): Model to use for labeling
        labels (List[str]): List of labels
        sample (float): The fraction of the dataset to label and use for filtering
        batch_size (int): Batch size for labeling
        num_workers (int): Number of workers for labeling
    """

    # Get a subset of the dataset
    subset = dataset.shuffle(seed=115).select(range(int(len(dataset) * sample)))

    # Label the subset
    subset = subset.map(
        lambda x: {"label": classify(x[text_column], labels, labeler_model)},
        batched=False,
        num_proc=num_workers,
    )

    return subset

# %% ../nbs/00_core.ipynb 7
def train_labeler(
    dataset,
    text_column,
    base_model_name,
    n_labels,
    training_args,
    num_workers=4,
    max_length=512,
    push_to_hub=True,
):
    """
    Trains a labeler model on a labeled dataset.

    Args:
        dataset (datasets.Dataset): Dataset to train on
        text_column (str): Name of the text column
        base_model_name (str): Name of the base model to use
        n_labels (int): Number of labels
        epochs (int): Number of epochs to train
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for training
        max_length (int): Maximum length of the input
    """
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, max_length=max_length)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name, num_labels=n_labels, max_length=max_length
    )
    model.config.id2label = {i: i for i in range(n_labels)}

    # Preprocess the dataset
    dataset = dataset.map(
        lambda x: tokenizer(
            x[text_column], padding="max_length", truncation=True, max_length=max_length
        ),
        batched=True,
        num_proc=num_workers,
    )

    # Split the dataset
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    # Get the data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Get the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Push the model to the hub
    if push_to_hub:
        trainer.push_to_hub()

    # Return the model
    return model, tokenizer

# %% ../nbs/00_core.ipynb 9
def filter_dataset(
    dataset, text_column, labeler_model, labels_to_keep, batch_size=32, num_workers=4
):
    """
    Filters a dataset using a labeler model.

    Args:
        dataset (datasets.Dataset): Dataset to filter
        text_column (str): Name of the text column
        labeler_model (transformers.pipelines.TextClassificationPipeline): Model to use for labeling
        labels_to_keep (list): List of labels to keep
        batch_size (int): Batch size for labeling
        num_workers (int): Number of workers for labeling
    """

    def label(x):
        predicted = labeler_model(x, padding=True, truncation=True, max_length=512)
        return {
            "label": [l["label"] for l in predicted],
            "score": [l["score"] for l in predicted],
        }

    # Label the dataset
    dataset = dataset.map(
        lambda x: label(x[text_column]),
        batched=True,
        batch_size=batch_size,
        num_proc=num_workers,
    )

    # Filter the dataset
    dataset = dataset.filter(lambda x: x["label"] in labels_to_keep)

    return dataset
