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

START_INDEX = 50000
END_INDEX = START_INDEX + NUM_SAMPLES


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
# LOAD BASE MODEL
# ==========================================================

def load_base_model():

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model.eval()

    return model


# ==========================================================
# LOAD LORA MODEL
# ==========================================================

def load_lora_model():

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
# EXTRACT YES / NO / MAYBE
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
# GENERATE MODEL RESPONSE
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

    generated_tokens = outputs[0][input_length:]

    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return response


# ==========================================================
# RUN COMPLETE MODEL
# ==========================================================

def evaluate_model(model, dataset, model_name):

    predictions = []

    responses = []

    print("\n")
    print("=" * 60)
    print(f"Evaluating {model_name}")
    print("=" * 60)

    start_time = time.time()

    for sample in tqdm(dataset):

        prompt = sample["prompt"]

        response = generate_response(model, prompt)

        prediction = extract_prediction(response)

        predictions.append(prediction)

        responses.append(response)

    total_time = time.time() - start_time

    avg_time = total_time / len(dataset)

    print(f"\nAverage inference time : {avg_time:.2f} sec/sample")

    return predictions, responses, avg_time


# ==========================================================
# FREE GPU MEMORY
# ==========================================================

def cleanup(model):

    del model

    gc.collect()

    torch.cuda.empty_cache()
import os

# ==========================================================
# CREATE OUTPUT DIRECTORY
# ==========================================================

os.makedirs(RESULT_FOLDER, exist_ok=True)

# ==========================================================
# EXTRACT GROUND TRUTH
# ==========================================================

ground_truth = []

for sample in dataset:

    response = sample["response"].lower()

    prediction = extract_prediction(response)

    ground_truth.append(prediction)

print(f"\nGround truth labels loaded : {len(ground_truth)}")


# ==========================================================
# BASE MODEL
# ==========================================================

print("\nLoading Base Model...\n")

base_model = load_base_model()

base_predictions, base_responses, base_time = evaluate_model(
    base_model,
    dataset,
    "Base Model"
)

cleanup(base_model)


# ==========================================================
# LORA MODEL
# ==========================================================

print("\nLoading Fine-tuned Model...\n")

lora_model = load_lora_model()

lora_predictions, lora_responses, lora_time = evaluate_model(
    lora_model,
    dataset,
    "Fine-tuned Model"
)

cleanup(lora_model)


# ==========================================================
# METRICS
# ==========================================================

labels = ["yes", "no", "maybe"]

base_accuracy = accuracy_score(
    ground_truth,
    base_predictions
)

lora_accuracy = accuracy_score(
    ground_truth,
    lora_predictions
)

base_precision, base_recall, base_f1, _ = precision_recall_fscore_support(
    ground_truth,
    base_predictions,
    labels=labels,
    average="weighted",
    zero_division=0
)

lora_precision, lora_recall, lora_f1, _ = precision_recall_fscore_support(
    ground_truth,
    lora_predictions,
    labels=labels,
    average="weighted",
    zero_division=0
)


# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame({

    "Prompt": dataset["prompt"],

    "Ground Truth": ground_truth,

    "Base Prediction": base_predictions,

    "Fine-tuned Prediction": lora_predictions,

    "Base Correct":
        [
            gt == pred
            for gt, pred in zip(
                ground_truth,
                base_predictions
            )
        ],

    "Fine-tuned Correct":
        [
            gt == pred
            for gt, pred in zip(
                ground_truth,
                lora_predictions
            )
        ],

    "Base Response": base_responses,

    "Fine-tuned Response": lora_responses,

})

csv_path = os.path.join(
    RESULT_FOLDER,
    "comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    ground_truth,
    lora_predictions,
    labels=labels
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)

fig, ax = plt.subplots(figsize=(6,6))

disp.plot(ax=ax)

plt.title("Fine-tuned Model Confusion Matrix")

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

metrics_file = os.path.join(
    RESULT_FOLDER,
    "metrics.txt"
)

with open(metrics_file, "w") as f:

    f.write("="*50+"\n")
    f.write("HealthGPT Evaluation\n")
    f.write("="*50+"\n\n")

    f.write("Base Model\n")

    f.write(f"Accuracy : {base_accuracy:.4f}\n")
    f.write(f"Precision: {base_precision:.4f}\n")
    f.write(f"Recall   : {base_recall:.4f}\n")
    f.write(f"F1 Score : {base_f1:.4f}\n")
    f.write(f"Avg Time : {base_time:.2f} sec\n\n")

    f.write("Fine-tuned Model\n")

    f.write(f"Accuracy : {lora_accuracy:.4f}\n")
    f.write(f"Precision: {lora_precision:.4f}\n")
    f.write(f"Recall   : {lora_recall:.4f}\n")
    f.write(f"F1 Score : {lora_f1:.4f}\n")
    f.write(f"Avg Time : {lora_time:.2f} sec\n\n")

    f.write(
        f"Accuracy Improvement : {(lora_accuracy-base_accuracy)*100:.2f}%\n"
    )


# ==========================================================
# PRINT REPORT
# ==========================================================

print("\n")
print("="*60)
print("HealthGPT Evaluation Report")
print("="*60)

print(f"\nSamples Evaluated : {len(dataset)}")

print("\nBase Model")

print(f"Accuracy : {base_accuracy:.4f}")
print(f"Precision: {base_precision:.4f}")
print(f"Recall   : {base_recall:.4f}")
print(f"F1 Score : {base_f1:.4f}")

print("\nFine-tuned Model")

print(f"Accuracy : {lora_accuracy:.4f}")
print(f"Precision: {lora_precision:.4f}")
print(f"Recall   : {lora_recall:.4f}")
print(f"F1 Score : {lora_f1:.4f}")

print("\nAccuracy Improvement")

print(f"{(lora_accuracy-base_accuracy)*100:.2f}%")

print("\nFiles Saved")

print(csv_path)
print(metrics_file)

print(
    os.path.join(
        RESULT_FOLDER,
        "confusion_matrix.png"
    )
)

print("\nEvaluation Complete!")