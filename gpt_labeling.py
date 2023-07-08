import os

from pydantic import BaseModel, Field

from datasets import concatenate_datasets, load_dataset
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.chat_models import AzureChatOpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from dotenv import load_dotenv
import time

from treasure_trove.core import label_dataset

load_dotenv(".env")
labels = ["high quality", "medium quality", "low quality"]
languages = ["python", "javascript"]
subsets = []
for lang in languages:
    ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}")["train"]
    sample = 50 / len(ds)
    subset = label_dataset(ds, "content", labels, sample=sample, num_workers=1)
    new_column = [lang] * len(subset)
    subset = subset.add_column("language", new_column)
    subsets.append(subset)

labeled_ds = concatenate_datasets(subsets)

# upload to huggingface
labeled_ds.push_to_hub("CarperAI/textbooks_A2YN_labeled_six_languages", private=True)

# print number of each class
print(f"Number of {labels[0]}: {len(labeled_ds.filter(lambda x: x['label'] == 0))}")
print(f"Number of {labels[1]}: {len(labeled_ds.filter(lambda x: x['label'] == 1))}")
print(f"Number of {labels[2]}: {len(labeled_ds.filter(lambda x: x['label'] == 2))}")
