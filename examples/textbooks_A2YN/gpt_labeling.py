import os

from datasets import concatenate_datasets, load_dataset
from squeakily.helpers import LLMLabeler
from treasure_trove.core import label_dataset

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

labels = ["high quality", "medium quality", "low quality"]
api_key = os.environ["OPENAI_KEY"]
labeler = LLMLabeler(instruction, labels, model_name="gpt-4", api_key=api_key) # gpt-3.5-turbo

languages = ["python", "go", "java", "javascript", "c", "c++"]
subsets = []
for lang in languages:
    ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}")["train"]
    sample = 50 / len(ds)
    subset = label_dataset(ds, "content", labeler, labels, sample=sample, num_workers=8)
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
