from datasets import load_from_disk

dataset = load_from_disk("data/processed")

print(dataset)

print(dataset[0])