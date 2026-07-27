# 🩺 HealthGPT

HealthGPT is a medical question-answering assistant built by fine-tuning **Llama 3.2 3B Instruct** using **QLoRA** on the **PubMedQA** dataset.

The project demonstrates an end-to-end LLM fine-tuning pipeline, including dataset preprocessing, tokenization, parameter-efficient fine-tuning, evaluation, and inference.

---

## Features

- Fine-tuned Llama 3.2 3B Instruct
- QLoRA (4-bit Quantization)
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes NF4 Quantization
- PubMedQA Dataset
- Gradient Checkpointing
- Mixed Precision (BF16)
- Optimized for consumer GPUs (RTX 4050 6GB)

---

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- BitsAndBytes
- Datasets
- Accelerate

---

## Project Structure

```text
HealthGPT/
│
├── scripts/
│   ├── preprocess.py
│   ├── tokenize.py
│   ├── add_labels.py
│   ├── train.py
│   └── inference.py
│
├── data/
│   ├── raw/
│   └── train_ready/
│
├── models/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Llama 3.2 3B Instruct |
| Fine-Tuning | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Optimizer | Paged AdamW 8-bit |
| Precision | BF16 |
| Dataset | PubMedQA |

---

## Dataset

- PubMedQA
- Biomedical research question-answering dataset containing real PubMed abstracts.

---

## Current Status

- Dataset preprocessing ✅
- Tokenization ✅
- QLoRA fine-tuning ✅
- Evaluation ✅
- Inference (Coming Soon)
- Web Interface (Coming Soon)

---

## Future Improvements

- Gradio Web UI
- RAG Integration
- Medical PDF Chat
- Hugging Face Deployment
- Docker Support
- Multi-turn Conversations

---

## License

This project is licensed under the MIT License.