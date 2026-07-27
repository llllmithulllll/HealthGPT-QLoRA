from datasets import load_from_disk

dataset = load_from_disk("data/tokenized")

def add_label(example):
    example["labels"]=example["input_ids"].copy()
    return example

train_dataset=dataset.map(add_label)
print(train_dataset[0].keys())
train_dataset.save_to_disk("data/train_ready")