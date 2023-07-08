import os
from pathlib import Path
from collections import Counter

from datasets import load_dataset

ds = load_dataset("roborovski/phi-1")["train"]
print(ds)
print(Counter(ds['label']))
print(Counter(ds['language']))
