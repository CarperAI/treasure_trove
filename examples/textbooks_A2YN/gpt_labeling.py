import os

from datasets import concatenate_datasets, load_dataset
from squeakily.helpers import LLMLabeler
from treasure_trove.core import label_dataset

# Number of highly educational: 20
# Number of medium educational: 93
# Number of low educational: 7

# Number of high quality: 40
# Number of medium quality: 73
# Number of low quality: 7

# Number of highly educational: 14
# Number of medium educational: 99
# Number of low educational: 7

instruction = f"""You are a senior level software engineer and you are tasked with reviewing a given code snippet's educational value. Use the following guidelines for determining the code's educational value:
Highly educational code has the following:
* Readability: The code is written in a way that is easy to understand and follow, with consistent detailed comments, formatting, meaningful variable names, and appropriate code structure.
* Modularity: The code is organized into reusable and independent modules or functions, making it easier to comprehend and reuse in other projects.
* Detailed explanations: The code is accompanied by thorough explanations of the concepts and techniques used, providing learners with a deeper understanding of the underlying principles.
* Good design principles: The code follows best practices for software design, such as encapsulation, separation of concerns, and adhering to design patterns, making it easier to understand and maintain.
Medium educational code has the following:
* Readability: The code is reasonably well-structured and readable, but there may be occasional inconsistencies, some comments, or less descriptive variable names.
* Partial modularity: The code contains some reusable components, but not all parts of the code are organized into separate modules or functions.
* Some explanations: The code may have limited explanations or comments that provide a general understanding of the code's logic and purpose.
* Adequate design principles: The code follows basic design principles, such as separation of concerns, but may not fully adhere to advanced design patterns or best practices.
Low educational code has the following:
* Poor readability: The code is poorly structured and difficult to follow, with little to no comments, inconsistent formatting and unclear variable names.
* No modularity: The code is written in a monolithic style, lacking any organization into reusable or independent modules or functions.
* Limited explanations: The code provides minimal or no explanations, leaving learners with little guidance on its logic or purpose.
* Neglects design principles: The code shows a lack of consideration for design principles, making it harder to comprehend, maintain, and extend.
* Boilerplate and autogenerated: The code contains a lot of boilerplate or autogenerated code, making it harder to comprehend and reuse.

Output nothing other than one of the following labels:
"""

labels = ["highly educational", "medium educational", "low educational"]
api_key = os.environ["OPENAI_KEY"]
labeler = LLMLabeler(instruction, labels, model_name="gpt-4", api_key=api_key) # gpt-3.5-turbo


languages = ["python", "go", "java", "javascript", "c", "cpp"]
subsets = []
for lang in languages:
    print(f"Labeling {lang}...")
    ds = load_dataset("CarperAI/starcoder_60k", data_dir=f"{lang}")["train"]
    sample_ratio = 1.0
    subset = label_dataset(ds, "cleaned_code", labeler, labels, sample=sample_ratio, num_workers=4)
    # write to parquet
    subset.to_parquet(f"data/{lang}.parquet")
    subsets.append(subset)

labeled_ds = concatenate_datasets(subsets)

# upload to huggingface
labeled_ds.push_to_hub("CarperAI/textbooks_A2YN_labeled_six_languages_60k", private=True)

# print number of each class
print(f"Number of {labels[0]}: {len(labeled_ds.filter(lambda x: x['label'] == 0))}")
print(f"Number of {labels[1]}: {len(labeled_ds.filter(lambda x: x['label'] == 1))}")
print(f"Number of {labels[2]}: {len(labeled_ds.filter(lambda x: x['label'] == 2))}")
