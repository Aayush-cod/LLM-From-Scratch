from datasets import load_dataset
import json

print("Downloading UltraChat dataset...")

# Load dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

print("Total samples:", len(dataset))

# ✅ Take only first 15k samples
dataset = dataset.select(range(4000))

print("Using subset:", len(dataset))

formatted_data = []

for sample in dataset:
    messages = sample["messages"]
    
    conversation_text = ""
    
    for msg in messages:
        if msg["role"] == "user":
            conversation_text += f"User: {msg['content']}\n"
        elif msg["role"] == "assistant":
            conversation_text += f"Assistant: {msg['content']}\n"

    formatted_data.append({
        "text": conversation_text.strip()
    })

# ✅ Save formatted dataset
with open("chat_sft_dataset.json", "w") as f:
    json.dump(formatted_data, f, indent=2)

print("✅ Saved 15k formatted conversations to chat_sft_dataset.json")