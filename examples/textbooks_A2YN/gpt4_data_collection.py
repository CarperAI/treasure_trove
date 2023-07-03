import os

from datasets import load_dataset
from squeakily.helpers import LLMLabeler
from transformers import pipeline, TrainingArguments
from treasure_trove.core import filter_dataset, label_dataset, train_labeler

instruction = """Please label the following code as either educational or non-educational.
Educational code is well written, follows best practices, has documentation such that it might be found in a textbook.
Non-educational code is poorly written, lacks documentation, contain bugs, or is not idiomatic.
Labels:
"""
labels = ["educational", "non-educational"]
api_key = os.environ["OPENAI_KEY"]
labeler = LLMLabeler(instruction, labels, model_name="gpt-4", api_key=api_key)

ds = load_dataset("bigcode/starcoderdata", data_dir="python")["train"]
sample = 100 / len(ds)
subset = label_dataset(ds, "content", labeler, labels, sample=sample)
# upload to huggingface
subset.push_to_hub("CarperAI/textbooks_A2YN_labeled", private=True)

# print number of each class
print(f"Number of {labels[0]}: {len(subset.filter(lambda x: x['label'] == 0))}")
print(f"Number of {labels[1]}: {len(subset.filter(lambda x: x['label'] == 1))}")
