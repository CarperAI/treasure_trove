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
from treasure_trove.core import LLMLabeler, instruction

load_dotenv(".env")
labels = ["high quality", "medium quality", "low quality"]
secondary_labels = ["high", "medium", "low"]
lang = "python"
max_chars = 4_096
num_workers = 8
labeler = LLMLabeler(
    instruction,
    labels,
    secondary_labels=secondary_labels,
)
res = labeler("def create()")
print(res)
dataset_chunks = []

buffer_size = 500
num_chunks = 100

print("Loading dataset..")
print("Loaded dataset.")

api_key = os.environ["OPENAI_KEY"]

max_failures = 5
failures = 0

ckpt_dir = "./checkpoints"
Path(ckpt_dir).mkdir(exist_ok=True)


def process(x):
    failures = 0
    total_cost = 0
    label_idx, cost_info = 0, {}
    while failures < max_failures:
        try:
            label_idx, cost_info = labeler(x["content"][:max_chars])
            time.sleep(1)
            break
        except Exception as e:
            failures += 1
            print(e)
            time.sleep(1)
    if cost_info:
        total_cost = cost_info["total_cost"]
        print(
            f"{label_idx} - tokens used: {cost_info['prompt_tokens']} | {cost_info['completion_tokens']} | {cost_info['total_cost']}"
        )
    else:
        print("row not classified.")
    return {"label": label_idx, "cost": total_cost}


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
    subset = subset.map(process, batched=False, num_proc=4)

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
