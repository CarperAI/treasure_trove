import os
import re
import ast
from datasets import concatenate_datasets, load_dataset
from treasure_trove.core_sk import label_dataset_sk, sk_function
import semantic_kernel as sk
import semantic_kernel.connectors.ai.open_ai as sk_oai
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from dotenv import load_dotenv
load_dotenv()

kernel = sk.kernel()

deployment, api_key, endpoint = sk.azure_openai_settings_from_dot_env()
kernel.add_text_completion_service(
    "code_eval", AzureChatCompletion(deployment, endpoint, api_key)
)

skills_dir = os.getcwd() + "/skills"
skill_category = "annotation"
func_name = "gpt_labeling"
skfunction = sk_function(skills_dir, skill_category, func_name)

skfunction_context = kernel.create_new_context()

labels = ["high quality", "medium quality", "low quality"]
languages = ["python", "go", "java", "javascript", "c", "c++"]
subsets = []
for lang in languages:
    ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}")["train"]
    sample = 50 / len(ds)
    subset = label_dataset_sk(ds, "content", labels, skfunction, sample=sample, num_workers=8)
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