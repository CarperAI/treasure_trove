import os
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, IterableDataset, Dataset
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
chunks_to_process = 20

print("Loading dataset..")
dataset = load_dataset("parquet", data_files={"train": "data-00000-of-00144.parquet"})[
    "train"
]
print("Loaded dataset.")

api_key = os.environ["OPENAI_KEY"]

subset_save_interval = 100

max_failures = 5
failures = 0

ckpt_dir = "./checkpoints"
Path(ckpt_dir).mkdir(exist_ok=True)


def process(x):
    failures = 0
    label_idx, cost_info = 0, {}
    while failures < max_failures:
        try:
            label, cost_info = labeler(x["content"][:max_chars])
            label_idx = labels.index(label)
            print(label, label_idx)
            time.sleep(1)
            break
        except Exception as e:
            failures += 1
            print(e)
            time.sleep(1)
    print(
        f"classified {i}: {label} - tokens used: {cost_info['prompt_tokens']} | {cost_info['completion_tokens']}"
    )
    return {"label": label_idx, "cost": cost_info["total_cost"]}


processed_chunk_datasets = []
start_idx = 1

for i in range(start_idx, start_idx + buffer_size, 1):
    print(f"Chunk {i} / {chunks_to_process + start_idx} starting...")

    subset = dataset[i : i + buffer_size]

    # Label the subset
    subset = dataset.map(process, batched=False, num_proc=8)

    processed_chunk_datasets.append(subset)

    all_datasets: Dataset = concatenate_datasets(processed_chunk_datasets)
    all_datasets.push_to_hub("roborovski/phi-1", private=True)
    all_datasets.to_parquet(
        os.path.join(
            ckpt_dir, f"processed_{start_idx}_to_{chunks_to_process+start_idx}"
        )
    )

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
