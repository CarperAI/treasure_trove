import re
import os

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)
import time
import openai

openai.api_key = os.getenv("OPENAI_KEY")


from typing import List

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
{0}
"""


class LLMLabeler:
    def __init__(
        self,
        instruction: str,
        labels: List[str],
        secondary_labels: List[str],
    ):
        self.instruction = instruction
        self.labels = labels
        self.secondary_labels = secondary_labels

    def find_label(self, text: str, labels: List[str]):
        for i, label in enumerate(labels):
            pattern = re.compile(re.escape(label), re.IGNORECASE | re.MULTILINE)
            match = re.search(pattern, text)
            if bool(match):
                return i
        return None

    def cost_info(self, oai_response):
        prompt_tokens = oai_response["usage"]["prompt_tokens"]
        completion_tokens = oai_response["usage"]["completion_tokens"]
        total_cost = 0.0015 * prompt_tokens + 0.0002 * completion_tokens

        return dict(
            total_cost=total_cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def __call__(self, text: str):
        formatted_instruction = instruction.format(self.labels)
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=0,
            max_tokens=4,
            messages=[
                {"role": "system", "content": formatted_instruction},
                {"role": "user", "content": text},
            ],
        )
        if "error" in completion:
            return 0, None
        output_text = completion["choices"][0]["message"]["content"]
        label_idx = self.find_label(output_text, self.labels)
        if not label_idx:
            label_idx = self.find_label(output_text, self.secondary_labels)
        cost_info = self.cost_info(completion)
        if not label_idx:
            raise Exception(f"Label not found in text: {output_text}")
        return label_idx, cost_info

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
