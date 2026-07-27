from transformers import AutoModelForCausalLM, AutoTokenizer

model_name="meta-llama/Llama-3.2-3B-Instruct"

save_path=".models/base/llama-3.2-3B-Instruct"

tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto")

print(f"Saving model to {save_path}")
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print("Done!")
