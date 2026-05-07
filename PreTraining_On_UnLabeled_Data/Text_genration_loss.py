import torch
from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel
import tiktoken
from PreTraining_On_UnLabeled_Data.Utility_function_for_text_toTokenId_conversion import token_ids_to_text

# GPT Configuration Dictionary
GPT_CONFIG_124M = {
    "vocab_size" :  50257,
    "context_length" : 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

tokenizer = tiktoken.get_encoding("gpt2")


# Utility functions for text to token ID conversion



# def token_ids_to_text(token_ids, tokenizer):
#     flat = token_ids.squeeze(0) #Removes batch dimension
#     return tokenizer.decode(flat.tolist())

# inputs 
inputs = torch.tensor([[16833, 3626, 6100], # "every effort moves"
                       [40, 1107, 588] #"I really like"
                       ])

# targets

targets = torch.tensor([[3626,6100, 345 ], #" effort moves you"
                        [1107, 588, 11311] #" really like chocolate"
                        ])


# we feed the inputs into the model to calculate logit vectors for the two input

with torch.no_grad():
        logits = model(inputs)

        
        # compute probability distribution
        probas = torch.softmax(logits, dim=-1) 
        print(probas.shape)
        # print(probas)


# applying argmax to find max probablity and its token id
        token_ids = torch.argmax(probas, dim=-1, keepdim = True)
        print("\n Token_ids: ", token_ids)


        print(f"\n Targets batch1: ", token_ids_to_text(targets[0],tokenizer))
        print(f"\n Outputs batch1: ", token_ids_to_text(token_ids[0].flatten(),tokenizer))


# Initial softmax probabilities scores corresponding to target tokens 
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("text 1: ", target_probas_1)

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("text 1: ", target_probas_2)


# Calculating loss  manually log
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)

# calculateing avergae log
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)

# turning negative log value to positive
neg_avg_loss_probas = avg_log_probas * -1
print(neg_avg_loss_probas)

# Calculating loss using Pytorch's Cross entropy loss
print("logits shape: ", logits.shape)
print("target shape", targets.shape)


# Flatten them by combinning over batch dimension

logits_flatten = logits.flatten(0,1)
targets_flatten = targets.flatten(0,1)
print("Logits_flatten: ", logits_flatten.shape)
print("target_flatten", targets_flatten.shape)

# using cross entropy loss
loss = torch.nn.functional.cross_entropy(logits_flatten, targets_flatten)
print("\nloss: ",loss)





