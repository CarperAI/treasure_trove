from typing import Optional, List

import fire
import re

from llama import Llama


instruction_simple = f"""Determine the following code's quality value for a software engineer whose goal is to improve their programming ability.
High quality code has the following:
* Readability: The code is written in a way that is easy to understand and follow.
* Modularity: The code is organized into reusable and independent modules or functions.
* Detailed explanations: The code is accompanied by explanations of the concepts used.
* Good design principles: The code follows best practices for software design.
Medium quality code has the following:
* Readability: The code is reasonably well-structured and readable.
* Partial modularity: The code contains some reusable components.
* Some explanations: The code may have limited explanations or comments.
* Adequate design principles: The code follows basic design principles.
Low quality code has the following:
* Poor readability: The code is poorly structured and difficult to follow.
* No modularity: The code is written in a monolithic style.
* Limited explanations: The code provides minimal or no explanations.
* Neglects design principles: The code shows a lack of consideration for design principles.

Output nothing other than one of the following labels:
High quality
Medium quality
Low quality
"""


def find_label(text: str, labels: List[str]):
    for i, label in enumerate(labels):
        pattern = re.compile(re.escape(label), re.IGNORECASE | re.MULTILINE)
        match = re.search(pattern, text)
        if bool(match):
            return i
    return None


import os
from pathlib import Path

from datasets import (
    concatenate_datasets,
    load_dataset,
    IterableDataset,
    Dataset,
    ReadInstruction,
)
from dotenv import load_dotenv

import time

load_dotenv(".env")
labels = ["high quality", "medium quality", "low quality"]
secondary_labels = ["high", "medium", "low"]
lang = "python"
max_chars = 4_096
num_workers = 8
dataset_chunks = []

buffer_size = 500
num_chunks = 100

print("Loading dataset..")
print("Loaded dataset.")

max_failures = 5
failures = 0

max_gen_len = 512
max_seq_len = 1024
temperature = 0.1
top_p = 0.2
max_batch_size = 4


ckpt_dir = "../llama/7Bf"
tokenizer_path = "../llama/tokenizer.model"

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=max_seq_len,
    max_batch_size=max_batch_size,
)


def process(x):
    total_cost = 0
    label_idx = 0
    dialogs = []
    for i in range(len(x["content"])):
        code_sample = x["content"][i][:max_gen_len]
        dialogs.append(
            [
                {"role": "system", "content": instruction_simple},
                {"role": "user", "content": code_sample},
            ]
        )
    results = generator.chat_completion(
        dialogs,  # type: ignore
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    batch_labels = []
    for i in range(len(dialogs)):
        completion_text = results[i]["generation"]["content"]
        label = find_label(completion_text, labels)
        batch_labels.append(label)
    return {"label": batch_labels}


processed_chunk_datasets = []

first_save_idx = 8000

for i in range(num_chunks):
    split = ReadInstruction(
        "train", from_=i * buffer_size, to=(i + 1) * buffer_size, unit="abs"
    )
    # if i < first_save_idx // buffer_size:
    #     print(f"skipping chunk {i}: {split}")
    #     continue
    print(f"processing chunk {i}: {split}")
    subset = load_dataset(
        "parquet", split=split, data_files={"train": "data-00000-of-00144.parquet"}
    )

    # Label the subset
    subset = subset.map(process, batched=True, batch_size=max_batch_size, num_proc=1)

    processed_chunk_datasets.append(subset)

    if i > first_save_idx // buffer_size:
        all_datasets: Dataset = concatenate_datasets(processed_chunk_datasets)
        try:
            all_datasets.push_to_hub("roborovski/phi-1", private=True)
            all_datasets.to_parquet(os.path.join(ckpt_dir, f"processed_{i}"))
        except Exception as e:
            print(e)

        # print number of each class
        print(
            f"Number of {labels[0]}: {len(all_datasets.filter(lambda x: x['label'] == 0))}"
        )
        print(
            f"Number of {labels[1]}: {len(all_datasets.filter(lambda x: x['label'] == 1))}"
        )
        print(
            f"Number of {labels[2]}: {len(all_datasets.filter(lambda x: x['label'] == 2))}"
        )
