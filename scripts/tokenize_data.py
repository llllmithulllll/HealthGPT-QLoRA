from datasets import load_from_disk
from transformers import AutoTokenizer

dataset=load_from_disk("data/processed")
model_name="meta-llama/Llama-3.2-3B-Instruct"
tokenizer=AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(examples):
    text=examples["prompt"]+ "\n\n" + examples["response"]
    return tokenizer(text, truncation=True, padding="max_length", max_length=512)
tokenized_dataset=dataset.map(tokenize)
print(tokenized_dataset[0].keys())
print(tokenized_dataset[0]["input_ids"][:20])
tokenized_dataset.save_to_disk("data/tokenized")
tokenizer.save_pretrained("models/tokenizer")