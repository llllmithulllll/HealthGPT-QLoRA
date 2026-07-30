from datasets import load_from_disk
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
import torch
from peft import PeftModel
from transformers import TrainingArguments, Trainer

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ======================================================
# Load Dataset
# ======================================================

dataset = load_from_disk("data/train_ready")

dataset = dataset.shuffle(seed=42)

# Second chunk (change this for future chunks)
dataset = dataset.select(range(50000,85000))

split = dataset.train_test_split(
    test_size=0.1,
    seed=42
)

train_dataset = split["train"]
val_dataset = split["test"]

# ======================================================
# Tokenizer
# ======================================================

tokenizer = AutoTokenizer.from_pretrained("models/tokenizer")
tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# Load Base Model
# ======================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    ".models/base/llama-3.2-3B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

model.gradient_checkpointing_enable()
model.config.use_cache = False

# ======================================================
# Load Existing LoRA
# ======================================================

print("Loading previous LoRA adapter...")

model = PeftModel.from_pretrained(
    model,
    "models/pubmed_lora",
    is_trainable=True
)

model.print_trainable_parameters()

# ======================================================
# Training Arguments
# ======================================================

training_args = TrainingArguments(
    output_dir="models/pubmed_lora",

    num_train_epochs=1,

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=120,

    bf16=True,

    optim="paged_adamw_8bit",

    max_grad_norm=1.0,

    logging_steps=20,

    eval_strategy="steps",
    eval_steps=1000,

    save_strategy="steps",
    save_steps=1000,

    save_total_limit=2,

    report_to="none",
)

# ======================================================
# Trainer
# ======================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

print("\nContinuing training...\n")

trainer.train()

trainer.save_model("models/pubmed_lora")

print("\nTraining Complete!\n")