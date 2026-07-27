from datasets import load_from_disk
from collections import Counter

dataset=load_from_disk("data/processed")
print(dataset)

missing_prompt=0
for examples in dataset:
    if not examples["prompt"].strip():
        missing_prompt+=1

print("Missing prompt:",missing_prompt)

missing_response=0
for examples in dataset:
    if not examples["response"].strip():
        missing_response+=1

print("Missing response:",missing_response)

wrong_prompt_type=0
wrong_response_type=0

for example in dataset:
    if not isinstance(example["prompt"], str):
        wrong_prompt_type+=1
    if not isinstance(example["response"], str):
        wrong_response_type+=1

print("Wrong prompt type:",wrong_prompt_type)
print("Wrong response type:",wrong_response_type)


prompts=dataset["prompt"]
duplicates=len(prompts)-len(set(prompts))
print("Duplicate prompts:",duplicates)

counts=Counter(dataset["prompt"])
for prompt,count in counts.items():
    if count>1:
        print("=== Duplicate prompt ===")
        print(f"appearances: {count}")
        print(prompt[:500])
        print()