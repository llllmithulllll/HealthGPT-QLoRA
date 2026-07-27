import json 
from datasets import Dataset

with open("data/ori_pqaa.json","r", encoding="utf-8") as f:
    data = json.load(f )

print(type(data))
print(len(data))

examples=list(data.values())

print(type(examples))
print(len(examples))    

dataset=Dataset.from_list(examples)
print(dataset)

def preprocess(examples):
    context="\n\n".join(examples["CONTEXTS"])

    prompt=f"""You are a helpful medical AI assistant.
    
    Question:{examples["QUESTION"]}

    Research context:{context}"""

    response=f"""Answer:{examples["final_decision"].capitalize()}
    Explanation:{examples["LONG_ANSWER"]}"""
    return {"prompt":prompt,"response":response}

processed_dataset = dataset.map(preprocess)
print(processed_dataset[0]["prompt"])
print("-" * 80)
print(processed_dataset[0]["response"])

preprocessed_dataset=processed_dataset.remove_columns([
    "QUESTION",
    "CONTEXTS",
    "LABELS",
    "LONG_ANSWER",
    "MESHES",
    "final_decision"
])

print(preprocessed_dataset)
print(preprocessed_dataset[0])
