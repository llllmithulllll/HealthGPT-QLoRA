# 🩺 HealthGPT

HealthGPT is a medical question-answering assistant built by fine-tuning **Llama 3.2 3B Instruct** using **QLoRA** on the **PubMedQA** dataset.

The project demonstrates a complete end-to-end Large Language Model (LLM) fine-tuning pipeline, covering everything from raw dataset preprocessing to inference on a custom medical assistant.

To overcome the limitations of training on a consumer GPU (RTX 4050 Laptop with 6GB VRAM), the model is trained incrementally on dataset chunks while continuously loading the previously trained LoRA adapter.

---

# ✨ Features

- Fine-tuned Llama 3.2 3B Instruct
- QLoRA (4-bit Quantization)
- PEFT (Parameter-Efficient Fine-Tuning)
- BitsAndBytes NF4 Quantization
- PubMedQA Dataset
- Gradient Checkpointing
- Mixed Precision (BF16)
- Incremental Chunk-Based Training
- Consumer GPU Optimized (RTX 4050 6GB)
- Medical Question Answering
- Local Inference Support

---

# 🛠 Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- BitsAndBytes
- Datasets
- Accelerate

---

# 📂 Project Structure

```text
HealthGPT/
│
├── scripts/
│   ├── preprocess.py
│   ├── tokenize.py
│   ├── add_labels.py
│   ├── train.py
│   ├── continue_training.py
│   ├── inference.py
│   └── evaluate.py
│
├── data/
│   ├── raw/
│   └── train_ready/
│
├── models/
│   ├── base/
│   │   └── llama-3.2-3B-Instruct/
│   │
│   ├── tokenizer/
│   │
│   └── pubmed_lora/
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | Llama 3.2 3B Instruct |
| Fine-Tuning | QLoRA |
| Quantization | 4-bit NF4 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| Optimizer | Paged AdamW 8-bit |
| Precision | BF16 |
| Batch Size | 1 |
| Gradient Accumulation | 8 |
| Sequence Length | 512 |
| Dataset | PubMedQA |

---

# 📚 Dataset

**PubMedQA**

A biomedical research question-answering dataset containing real PubMed abstracts.

The dataset is preprocessed into instruction-following prompt-response pairs before tokenization and training.

---

# 🚀 Training Pipeline

The complete workflow is:

```text
Raw PubMedQA
        │
        ▼
Preprocessing
        │
        ▼
Tokenization
        │
        ▼
Label Generation
        │
        ▼
QLoRA Fine-Tuning
        │
        ▼
Evaluation
        │
        ▼
Inference
```

---

# 🔄 Incremental Training

Since the project is trained on a **6GB RTX 4050 Laptop GPU**, the complete dataset cannot be efficiently trained in a single run.

Instead, the model is trained incrementally.

Example:

```text
Chunk 1

0      → 35,000

↓

Chunk 2

35,000 → 70,000

↓

Chunk 3

70,000 → 105,000

↓

Chunk 4

105,000 → 140,000

↓

Chunk 5

140,000 → 175,000

↓

Chunk 6

175,000 → 210,000
```

The first chunk is trained using:

```bash
python scripts/train.py
```

Every subsequent chunk uses:

```bash
python scripts/continue_training.py
```

The continuation script loads the previously trained LoRA adapter and continues learning on the next dataset chunk.

---

# 🧠 Training Strategy

Instead of retraining from scratch, HealthGPT loads the existing LoRA adapter before each new training phase.

This allows the model to continually improve while keeping GPU memory usage low.

```python
model = PeftModel.from_pretrained(
    model,
    "models/pubmed_lora",
    is_trainable=True
)
```

This approach preserves everything learned from previous chunks while training on new data.

---

# 📈 Current Progress

- ✅ Dataset preprocessing
- ✅ Prompt engineering
- ✅ Tokenization
- ✅ Label generation
- ✅ QLoRA fine-tuning
- ✅ Incremental chunk-based training
- ✅ Local inference
- ✅ Evaluation pipeline

---

# 🎯 Future Improvements

- Gradio Web UI
- Retrieval-Augmented Generation (RAG)
- Medical PDF Chat
- Hugging Face Deployment
- Docker Support
- Multi-turn Medical Conversations
- Benchmark against Base Llama
- LoRA Merge for Standalone Model
- Additional Medical Datasets (MedQA, MedMCQA)

---

# 📄 License

This project is licensed under the MIT License.