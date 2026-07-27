import json
from datasets import Dataset

def load_dataset(path="data/ori_pqaa.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = list(data.values())
    return Dataset.from_list(examples)