from transformers import AutoTokenizer
import transformers
import torch

model = "../llama-7bf-hf"

instruction_simple = f"""Determine the following code's quality value for a software engineer whose goal is to improve their programming ability.
High quality code has the following:
* Readability: The code is written in a way that is easy to understand and follow.
* Modularity: The code is organized into reusable and independent modules or functions.
* Detailed explanations: The code is accompanied by explanations of the concepts used.
* Good design principles: The code follows best practices for software design.
Medium quality code has the following:
* Readability: The code is reasonably well-structured and readable.
* Partial modularity: The code contains some reusable components.
* Some explanations: The code may have limited explanations or comments.
* Adequate design principles: The code follows basic design principles.
Low quality code has the following:
* Poor readability: The code is poorly structured and difficult to follow.
* No modularity: The code is written in a monolithic style.
* Limited explanations: The code provides minimal or no explanations.
* Neglects design principles: The code shows a lack of consideration for design principles.

Output nothing other than one of the following labels:
High quality
Medium quality
Low quality
"""


tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "conversational",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    instruction_simple,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

