import time
import os

import numpy as np

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import time


from pydantic import BaseModel, Field

from datasets import concatenate_datasets, load_dataset
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate

class LLMLabelerParser(BaseModel):
    labels: List = Field(
        ..., title="Labels", description="Labels that the LLM classifies the text as"
    )


class LLMLabeler:
    def __init__(
        self,
        instruction: str,
        labels: List,
        model_name: str = "gpt-3.5-turbo",
        api_key: str = None,
        model_type: str = "openai",
    ):
        self.instruction = instruction
        self.labels = labels
        # Set up a parser + inject instructions into the prompt template.
        self.parser = PydanticOutputParser(pydantic_object=LLMLabelerParser)
        prompt = PromptTemplate(
            template="{instruction}\n{labels}\n{format_instructions}\n",
            input_variables=["instruction", "labels"],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )
        system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)
        human_template = "{text}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        self.chat_template = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )

        if model_type == "azure":
            raise NotImplementedError("Azure models are not supported yet")
        elif model_type == "openai":
            self.model = ChatOpenAI(
                openai_api_key=api_key, model_name=model_name, temperature=0
            )
        else:
            raise ValueError(f"Model type {model_type} is not supported")

    def __call__(self, text: str):
        messages = self.chat_template.format_prompt(
            instruction=self.instruction, labels=self.labels, text=text
        ).to_messages()
        output = self.model(messages)
        print('model output', output.content)
        if output.content in self.labels:
            return [output]
        predicted_labels = self.parser.parse(output.content)
        print('pred labels', predicted_labels)
        # check if all the predicted tags are in the list of tags
        assert all(
            [label in self.labels for label in predicted_labels.labels]
        ), f"Predicted labels {predicted_labels.labels} are not in the list of tags {self.labels}"
        return predicted_labels.labels


instruction = f"""Determine the following code's quality value for a software engineer whose goal is to improve their programming ability.
High quality code has the following:
* Readability: The code is written in a way that is easy to understand and follow, with consistent detailed comments, formatting, meaningful variable names, and appropriate code structure.
* Modularity: The code is organized into reusable and independent modules or functions, making it easier to comprehend and reuse in other projects.
* Detailed explanations: The code is accompanied by thorough explanations of the concepts and techniques used, providing learners with a deeper understanding of the underlying principles.
* Good design principles: The code follows best practices for software design, such as encapsulation, separation of concerns, and adhering to design patterns, making it easier to understand and maintain.
Medium quality code has the following:
* Readability: The code is reasonably well-structured and readable, but there may be occasional inconsistencies, some comments, or less descriptive variable names.
* Partial modularity: The code contains some reusable components, but not all parts of the code are organized into separate modules or functions.
* Some explanations: The code may have limited explanations or comments that provide a general understanding of the code's logic and purpose.
* Adequate design principles: The code follows basic design principles, such as separation of concerns, but may not fully adhere to advanced design patterns or best practices.
Low quality code has the following:
* Poor readability: The code is poorly structured and difficult to follow, with little to no comments, inconsistent formatting and unclear variable names.
* No modularity: The code is written in a monolithic style, lacking any organization into reusable or independent modules or functions.
* Limited explanations: The code provides minimal or no explanations, leaving learners with little guidance on its logic or purpose.
* Neglects design principles: The code shows a lack of consideration for design principles, making it harder to comprehend, maintain, and extend.

Output nothing other than one of the following labels:
"""


def classify(x, labels, max_failures=5, default_label=0):
    failures = 0
    api_key = os.environ["OPENAI_KEY"]
    labeler = LLMLabeler(
        instruction, labels, model_name="gpt-3.5-turbo", api_key=api_key
    )

    while failures < max_failures:
        try:
            label = labeler(x)[0]
            label_idx = labels.index(label)
            print(label, label_idx)
            return label_idx
        except Exception as e:
            failures += 1
            print(e)
            time.sleep(1)
            pass
    if failures == max_failures:
        return default_label


def label_dataset(
    dataset,
    text_column,
    labels,
    sample=0.1,
    num_workers=4,
    max_chars=4_096,
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
        max_chars (int): Maximum number of characters to truncate the text to before labeling (reduces rate limiting errors)
    """

    # Get a subset of the dataset
    subset = dataset.shuffle(seed=115).select(range(int(len(dataset) * sample)))

    # Label the subset
    subset = subset.map(
        lambda x: {
            "label": classify(x[text_column][:max_chars], labels)
        },
        batched=False,
        num_proc=num_workers,
    )

    return subset


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

    # Get the trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
    )

    # Train the model
    trainer.train()

    # Push the model to the hub
    if push_to_hub:
        trainer.push_to_hub()

    # Return the model
    return model, tokenizer


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
