import os
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from dotenv import load_dotenv

from treasure_trove.core import classify

load_dotenv(".env")
labels = ["high quality", "medium quality", "low quality"]
lang = "python"
processed_subsets = []
max_chars = 4_096
num_workers = 8
epochs = 2
buffer_size = 1000
dataset = load_dataset(
    "bigcode/the-stack-dedup", data_dir=f"data/{lang}", streaming=True
)["train"]
subset = dataset.shuffle(seed=115, buffer_size=buffer_size)

for epoch in range(epochs):
    subset.set_epoch(epoch)

    procesed = subset.map(
        lambda x: {"label": classify(x["content"][:max_chars], labels)},
        batched=False,
    )

    lang_column = [lang] * buffer_size
    procesed = procesed.add_column("language", lang_column)
    processed_subsets.append(procesed)

    processed_ds = concatenate_datasets(processed_subsets)

    # upload to huggingface
    ckpt_dir = "./checkpoints"
    Path(ckpt_dir).mkdir(exist_ok=True)
    processed_ds.save_to_disk(ckpt_dir + "/latest")
    processed_ds.push_to_hub("roborovski/phi-1", private=True)

    # print number of each class
    print(f"Number of {labels[0]}: {len(processed_ds.filter(lambda x: x['label'] == 0))}")
    print(f"Number of {labels[1]}: {len(processed_ds.filter(lambda x: x['label'] == 1))}")
    print(f"Number of {labels[2]}: {len(processed_ds.filter(lambda x: x['label'] == 2))}")
