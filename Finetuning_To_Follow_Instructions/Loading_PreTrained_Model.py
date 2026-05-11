from gpt_download import download_and_load_gpt2
from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel
from Loading_OpenAI_weights_GPTModel import load_weights_into_gpt
import torch
from Finetuning_To_Follow_Instructions.Dataset_preparation import format_input
from Finetuning_To_Follow_Instructions.Dataset_preparation import val_data
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

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

torch.manual_seed(123)

input_text = format_input(val_data[0])
print(input_text)

from PreTraining_On_UnLabeled_Data.Modifying_text_generate_function_08 import generate
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import text_to_token_ids
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import token_ids_to_text

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
print(generated_text)

response_text = (
    generated_text[len(input_text):]
    .replace("### Response:", "")
    .strip()
)
print(response_text)





# ============================================================
# 7.6 Finetuning the LLM on instruction data
# ============================================================

from PreTraining_On_UnLabeled_Data.Training_An_LLM_04 import train_model_simple
from PreTraining_On_UnLabeled_Data.Traning_Validation_Dataset_Losses_03 import calc_loss_loader
from Finetuning_To_Follow_Instructions.Dataloader import device
from Finetuning_To_Follow_Instructions.Dataloader import train_loader
from Finetuning_To_Follow_Instructions.Dataloader import val_loader
from Finetuning_To_Follow_Instructions.Dataloader import test_loader

model.to(device)

# ✅ Path where finetuned model will be saved
finetuned_model_path = f"instruction_finetuned_{model_size}.pth"

if os.path.exists(finetuned_model_path):
    print("✅ Finetuned model found. Loading weights...")
    model.load_state_dict(torch.load(finetuned_model_path, map_location=device))
    model.eval()

else:
    print("🚀 No finetuned model found. Starting instruction finetuning...")

    torch.manual_seed(123)

    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("Training loss before finetuning:", train_loss)
    print("Validation loss before finetuning:", val_loss, "\n")

    import time
    start_time = time.time()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    num_epochs = 2

    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=5,
        eval_iter=5,
        start_context=format_input(val_data[0]),
        tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60

    print(f"\nTraining completed in {execution_time_minutes:.2f} minutes.")

    # ✅ Save finetuned model
    torch.save(model.state_dict(), finetuned_model_path)
    print("✅ Finetuned model saved.")