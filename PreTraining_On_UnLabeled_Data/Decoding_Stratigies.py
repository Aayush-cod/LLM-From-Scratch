import torch
import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion import token_ids_to_text
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion import text_to_token_ids
from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import generate_text_simple


from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel
from PreTraining_On_UnLabeled_Data.Training_An_LLM import GPT_CONFIG_124M


model = GPTModel(GPT_CONFIG_124M)
# Load trained weights
model.load_state_dict(torch.load("gpt_model.pth", map_location="cpu"))
model.eval()


token_ids = generate_text_simple(
    model = model,
    idx = text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=25,
    context_size= GPT_CONFIG_124M["context_length"]


)

print("Output text: \n", token_ids_to_text(token_ids, tokenizer))



