from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

import torch
import pandas as pd
import re
import time
import gc
import os

from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt

# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_MODEL = ".models/base/llama-3.2-3B-Instruct"
LORA_MODEL = "models/pubmed_lora"

TOKENIZER = "models/tokenizer"

DATASET = "data/train_ready"

RESULT_FOLDER = "evaluation_results"

NUM_SAMPLES = 500

START_INDEX = 100000
END_INDEX = START_INDEX + NUM_SAMPLES

# Base model metrics (evaluate ONCE and keep these)

BASE_ACCURACY = 0.8060
BASE_PRECISION = 0.9613
BASE_RECALL = 0.8060
BASE_F1 = 0.8767

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("Loading evaluation dataset...")
print("=" * 60)

dataset = load_from_disk(DATASET)

dataset = dataset.shuffle(seed=42)

dataset = dataset.select(range(START_INDEX, END_INDEX))

print(f"Loaded {len(dataset)} evaluation samples.")

# ==========================================================
# TOKENIZER
# ==========================================================

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

tokenizer.pad_token = tokenizer.eos_token

# ==========================================================
# QUANTIZATION CONFIG
# ==========================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# ==========================================================
# LOAD LORA MODEL
# ==========================================================

def load_model():

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(
        base,
        LORA_MODEL
    )

    model.eval()

    return model

# ==========================================================
# EXTRACT LABEL
# ==========================================================

def extract_prediction(text):

    text = text.lower()

    patterns = [
        r"answer:\s*(yes|no|maybe)",
        r"\b(yes|no|maybe)\b"
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            return match.group(1)

    return "unknown"

# ==========================================================
# GENERATE RESPONSE
# ==========================================================

def generate_response(model, prompt):

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][input_length:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

# ==========================================================
# EVALUATE MODEL
# ==========================================================

predictions = []
responses = []

ground_truth = []

for sample in dataset:

    ground_truth.append(
        extract_prediction(sample["response"])
    )

print("\nLoading Fine-tuned Model...\n")

model = load_model()

start = time.time()

for sample in tqdm(dataset):

    response = generate_response(
        model,
        sample["prompt"]
    )

    responses.append(response)

    predictions.append(
        extract_prediction(response)
    )

avg_time = (time.time() - start) / len(dataset)

del model
gc.collect()
torch.cuda.empty_cache()

# ==========================================================
# METRICS
# ==========================================================

labels = ["yes", "no", "maybe"]

accuracy = accuracy_score(
    ground_truth,
    predictions
)

precision, recall, f1, _ = precision_recall_fscore_support(
    ground_truth,
    predictions,
    labels=labels,
    average="weighted",
    zero_division=0,
)

# ==========================================================
# SAVE CSV
# ==========================================================

os.makedirs(RESULT_FOLDER, exist_ok=True)

df = pd.DataFrame({

    "Prompt": dataset["prompt"],

    "Ground Truth": ground_truth,

    "Prediction": predictions,

    "Correct": [
        gt == pred
        for gt, pred in zip(
            ground_truth,
            predictions
        )
    ],

    "Response": responses,
})

csv_path = os.path.join(
    RESULT_FOLDER,
    "comparison.csv"
)

df.to_csv(csv_path, index=False)

# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    ground_truth,
    predictions,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(ax=ax)

plt.title("HealthGPT Confusion Matrix")

plt.savefig(
    os.path.join(
        RESULT_FOLDER,
        "confusion_matrix.png"
    )
)

plt.close()

# ==========================================================
# METRICS FILE
# ==========================================================

metrics_path = os.path.join(
    RESULT_FOLDER,
    "metrics.txt"
)

with open(metrics_path, "w") as f:

    f.write("="*50+"\n")
    f.write("HealthGPT Evaluation\n")
    f.write("="*50+"\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"Avg Time : {avg_time:.2f} sec\n\n")

    f.write(
        f"Accuracy Improvement : {(accuracy-BASE_ACCURACY)*100:.2f}%\n"
    )

# ==========================================================
# REPORT
# ==========================================================

print("\n")
print("="*60)
print("HealthGPT Evaluation Report")
print("="*60)

print(f"\nSamples Evaluated : {len(dataset)}")

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print(f"\nAccuracy Improvement : {(accuracy-BASE_ACCURACY)*100:.2f}%")

print(f"\nAverage Inference : {avg_time:.2f} sec/sample")

print("\nFiles Saved")

print(csv_path)
print(metrics_path)
print(os.path.join(
    RESULT_FOLDER,
    "confusion_matrix.png"
))

print("\nEvaluation Complete!")