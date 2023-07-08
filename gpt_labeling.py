import os
from pathlib import Path

from datasets import concatenate_datasets, load_dataset, IterableDataset, Dataset
from dotenv import load_dotenv

import time
from treasure_trove.core import LLMLabeler, instruction

load_dotenv(".env")
api_key = os.environ["OPENAI_KEY"]
labels = ["high quality", "medium quality", "low quality"]
lang = "python"
processed_subsets = []
max_chars = 4_096
num_workers = 8
labeler = LLMLabeler(instruction, labels, model_name="gpt-3.5-turbo", api_key=api_key)
res = labeler("def create()")
print(res)

buffer_size = 10_000
chunks_to_process = 10

print("Loading dataset..")
dataset = load_dataset(
    "bigcode/the-stack-dedup",
    data_dir=f"data/{lang}",
    streaming=True,
)["train"]
print("Loaded dataset.")

subset = dataset.shuffle(seed=100, buffer_size=buffer_size)

subset_save_interval = 100

total_cost = 0
max_failures = 5
failures = 0

ckpt_dir = "./checkpoints"
Path(ckpt_dir).mkdir(exist_ok=True)

for chunk in range(chunks_to_process):
    print(f"Chunk {chunk} / {chunks_to_process} starting...")

    processed_rows = []
    subset.set_epoch(chunk)

    for i, x in enumerate(subset):
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
        if failures != max_failures:
            total_cost += cost_info["total_cost"]
            print(
                f"classified {i}: {label} - tokens used: {cost_info['prompt_tokens']} | {cost_info['completion_tokens']}"
            )
            processed_rows.append({**x, "label": label, "language": lang})
        else:
            print(f"Max failures hit on idx {i}, continuing.")
        if i % subset_save_interval == 0:
            subset_ds = Dataset.from_list(processed_rows)
            subset_ds.save_to_disk(os.path.join(ckpt_dir, f"chunk_{chunk}_subset_{i}"))
            subset_ds.push_to_hub("roborovski/phi-1", private=True)

    processed_subsets.append(processed_rows)
    # Save all processed data
    all_datasets: Dataset = concatenate_datasets(processed_subsets)
    all_datasets.save_to_disk(os.path.join(ckpt_dir, "latest"))
    all_datasets.push_to_hub("roborovski/phi-1", private=True)

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
    print(f"Cost so far: {total_cost}")
