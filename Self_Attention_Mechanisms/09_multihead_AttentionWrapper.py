import torch
import torch.nn as nn

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1]

d_in = inputs.shape[1]
d_out = 2

batch = torch.stack((inputs, inputs), dim= 0)


class CasualAttention(nn.Module):
    def __init__(self,d_in, d_out,context_length,dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in,d_out, bias = qkv_bias) 
        self.W_key = nn.Linear(d_in,d_out, bias = qkv_bias) 
        self.W_value = nn.Linear(d_in,d_out, bias = qkv_bias) 
        self.dropout = nn.Dropout(dropout)

        # Here register_buffer is only a model part not used for trainable . It moves to GPU with the model,It gets saved with the model ,It is NOT updated by gradients
        self.register_buffer('mask',
                             torch.triu(torch.ones(context_length,context_length), diagonal=1))


    def forward(self, x):
        # 3D tensor[2,6,3- batches(b), rows or tokens(num_tokens), dimensionality(d_in) of each word]
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries =  self.W_query(x)
        values = self.W_value(x)

        # Here while masking one mask (6,6) is used for both 0 batch , 1 batch, and num tokens ensure that mask matches shape with attn scores ass context length may be 10,10 but input may be 2,6,3 so with num tokens it adjust mask to 6,6

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens],-torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights @ values
        return(context_vec)
    

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self,d_in, d_out,context_length,dropout,num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out,context_length,dropout, qkv_bias)
                                    for _ in range(num_heads)])
        
    def forward(self,x):
        return torch.cat([head(x) for head in self.heads], dim = -1)

    
torch.manual_seed(123)
context_length = batch.shape[1]
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads= 2)
context_vecs= mha(batch)

print(context_vecs)
print("Context_vecs.shape: ", context_vecs.shape)