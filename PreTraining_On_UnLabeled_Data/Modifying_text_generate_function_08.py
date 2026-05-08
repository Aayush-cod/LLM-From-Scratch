import torch
import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import token_ids_to_text
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion_01 import text_to_token_ids


from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel
from PreTraining_On_UnLabeled_Data.Training_An_LLM_04 import GPT_CONFIG_124M


model = GPTModel(GPT_CONFIG_124M)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load trained weights
model.load_state_dict(torch.load("gpt_model.pth", map_location="cpu"))
model.eval()

# Modified Text generation function
def generate(model, idx, max_new_tokens, context_size, 
             temperature = 1.0, top_k = None, eos_id = None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits , _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')),
                logits
            )    
            


        if temperature > 0.0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim = -1)
                idx_next = torch.multinomial(probs, num_samples=1)
        else:
                idx_next = torch.argmax(probs, dim =-1, keepdim=True)
        # It tells model when to stop generating like a fullstop
        if idx_next == eos_id:
                break
        
        idx = torch.cat((idx , idx_next), dim=1)

    return idx


# initialize
torch.manual_seed(123)
token_ids = generate(
    model = model,
    idx = text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=15,
    context_size= GPT_CONFIG_124M["context_length"],
    top_k=25,
    temperature=1.4


)

print("Output text: \n", token_ids_to_text(token_ids, tokenizer))


        