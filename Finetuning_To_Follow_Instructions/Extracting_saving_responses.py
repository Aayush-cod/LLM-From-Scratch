import torch
from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel
from Finetuning_To_Follow_Instructions.Dataset_preparation import test_data
from Finetuning_To_Follow_Instructions.Dataset_preparation import format_input
from PreTraining_On_UnLabeled_Data.Modifying_text_generate_function_08 import generate
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import text_to_token_ids
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import token_ids_to_text
from Finetuning_To_Follow_Instructions.Dataloader import device
import tiktoken
tokenizer = tiktoken.get_encoding("gpt2")
import os

BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

# device = torch.device("cpu")


model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("instruction_finetuned_355M.pth"))
model.eval()
model.to(device)

torch.manual_seed(123)


for entry in test_data[:3]:

    input_text = format_input(entry)

    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = (
        generated_text[len(input_text):]
        .replace("### Response:", "")
        .strip()
)

    print(input_text)
    print(f"\nCorrect response:\n>> {entry['output']}")
    print(f"\nModel response:\n>> {response_text.strip()}")
    print("-------------------------------------")

# Listing 7.9 Generating test set responses

from tqdm import tqdm
import json
output_json_path = "instruction-data-with-response.json"

if os.path.exists(output_json_path):
    print("✅ JSON file already exists. Loading existing responses...")

    with open(output_json_path, "r") as file:
        test_data_with_responses = json.load(file)

else:
    print("🚀 JSON file not found. Generating responses...")

    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )

        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )

        test_data[i]["model_response"] = response_text

    # ✅ Save file
    with open(output_json_path, "w") as file:
        json.dump(test_data, file, indent=4)

    print("✅ JSON file created and saved.")

    test_data_with_responses = test_data

print(test_data_with_responses[4])


import re


file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
torch.save(model.state_dict(), file_name)
print(f"Model saved as {file_name}")

# Load model via
# model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))

