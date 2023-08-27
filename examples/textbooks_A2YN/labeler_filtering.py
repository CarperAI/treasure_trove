from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

MODEL_NAME="CarperAI/code_edu_classifier_multi_lang"
TOKENIZER_NAME="bigcode/starencoder"

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
tokenizer.pad_token = tokenizer.eos_token
pipe = pipeline(
    "text-classification", model=MODEL_NAME, tokenizer=tokenizer, device="cuda:0"
)
data_dir = "<path to data>"
languages = ["python", "java", "javascript", "go", "c", "cpp"]
tokenizer_kwargs = {'padding':True,'truncation':True,'max_length':1024}

def func(x):
    labels = []
    scores = []
    for i in pipe(x["content"], truncation=True, padding="max_length", max_length=1024, batch_size=256):
        labels.append(i["label"])
        scores.append(i["score"])
    return {"label": labels, "score": scores}

for lang in languages:
    ds = load_dataset("parquet", data_dir=f"{data_dir}/{lang}", split="train")
    print(f"Loaded {lang} dataset with {len(ds)} examples")
    ds = ds.map(lambda x: func(x), batched=True, batch_size=256)
    ds.to_parquet(f"{data_dir}/{lang}_labeled/")