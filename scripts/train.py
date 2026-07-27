from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import (AutoModelForCausalLM, BitsAndBytesConfig)
import torch
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments,Trainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

dataset = load_from_disk("data/train_ready")

dataset=dataset.shuffle(seed=42)
dataset=dataset.select(range(35000))


split=dataset.train_test_split(
    test_size=0.1,
    seed=42
)
train_dataset=split["train"]
val_dataset=split["test"]

tokenizer=AutoTokenizer.from_pretrained("models/tokenizer")
tokenizer.pad_token = tokenizer.eos_token

bnb_config=BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model=AutoModelForCausalLM.from_pretrained(
    ".models/base/llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto"
)

# Enable gradient checkpointing

model.gradient_checkpointing_enable()
model.config.use_cache = False

lora_config=LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM")
model=get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir="models/pubmed_lora",

    num_train_epochs=1,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,          # changed from 2e-4

    bf16=True,

    optim="paged_adamw_8bit",     # NEW

    max_grad_norm=1.0,            # NEW

    logging_steps=20,

    eval_strategy="steps",
    eval_steps=500,

    save_strategy="steps",
    save_steps=500,

    save_total_limit=2,

    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)
trainer.train()
trainer.save_model("models/pubmed_lora")