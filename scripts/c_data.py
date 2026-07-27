from datasets import load_from_disk

dataset = load_from_disk("data/train_ready")

for i in range(len(dataset)):
    if len(dataset[i]["input_ids"]) != len(dataset[i]["labels"]):
        print("Mismatch:", i)

    if len(dataset[i]["input_ids"]) == 0:
        print("Empty:", i)