from Finetuning_For_Classification.Dataset_Dataloader import train_dataset
import torch

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

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

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"`max_length={BASE_CONFIG['context_length']}`"
)

# Preparing Model

from gpt_download import download_and_load_gpt2
from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel
from Loading_OpenAI_weights_GPTModel import load_weights_into_gpt


model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")




import tiktoken


tokenizer = tiktoken.get_encoding("gpt2")
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import token_ids_to_text
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import text_to_token_ids
from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import generate_text_simple

if __name__ == "__main__":
    
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    torch.save(model.state_dict(), "Pretrain_weight_model.pth")
    model.eval();

    text_1 = "Every effort moves you"

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_1, tokenizer),
        max_new_tokens=15,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))


    text_2 = (
        "Is the following text 'spam'? Answer with 'yes' or 'no':"
        " 'You are a winner you have been specially"
        " selected to receive $1000 cash or a $2000 award.'"
    )

    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(text_2, tokenizer),
        max_new_tokens=23,
        context_size=BASE_CONFIG["context_length"]
    )

    print(token_ids_to_text(token_ids, tokenizer))