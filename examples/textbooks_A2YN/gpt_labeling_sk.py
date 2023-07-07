import os
import re
import ast
from datasets import concatenate_datasets, load_dataset
from squeakily.helpers import LLMLabeler
from treasure_trove.core import label_dataset

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
annotation_skills = kernel.import_semantic_skill_from_directory(skills_dir, "annotation")
annotation_function = annotation_skills["gpt_labeling"]

annotation_context = kernel.create_new_context()

# Single example without function, should be integrated with label_dataset and classify
languages = ["python", "go", "java", "javascript", "c", "c++"]
subsets = []
for lang in languages:
    annotation_context['code'] 
    ds = load_dataset("bigcode/the-stack-smol", data_dir=f"data/{lang}")["train"]
    sample = 50 / len(ds)
    ds.shuffle(seed=115).select(range(int(len(ds) * sample)))
    evalutation = annotation_function(context=annotation_context)
    evaluation = re.sub('\s+', ' ', evaluation.result).replace('\n', '')
    eval_ast = ast.literal_eval(evaluation)
    rationale = eval_ast['rationale']
    label = eval_ast['evaluation']
    #....