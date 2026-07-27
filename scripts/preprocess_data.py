from load_data import load_dataset

dataset = load_dataset()

def preprocess(example):
    context = "\n\n".join(example["CONTEXTS"])

    prompt = f"""You are a helpful medical AI assistant.

Question:
{example["QUESTION"]}

Research Context:
{context}
"""

    response = f"""Answer: {example["final_decision"].capitalize()}

Explanation:
{example["LONG_ANSWER"]}
"""

    return {
        "prompt": prompt,
        "response": response
    }

processed_dataset = dataset.map(preprocess)

processed_dataset = processed_dataset.remove_columns([
    "QUESTION",
    "CONTEXTS",
    "LABELS",
    "LONG_ANSWER",
    "MESHES",
    "final_decision",
])

processed_dataset.save_to_disk("data/processed")