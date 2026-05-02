import torch
import torch.nn as nn
import tiktoken 

# Masked Multi head self attention
class MultiHeadAttention(nn.Module):
    def __init__(self,d_in, d_out,context_length,dropout,num_heads, qkv_bias = False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in,d_out, bias = qkv_bias) 
        self.W_key = nn.Linear(d_in,d_out, bias = qkv_bias) 
        self.W_value = nn.Linear(d_in,d_out, bias = qkv_bias) 
        self.out_proj = nn.Linear(d_out, d_out)
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

        # splits features or d_in into num_heads and head_dim

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1,2)
        queries= queries.transpose(1,2)
        values = values.transpose(1,2)




        # Here while masking one mask (6,6) is used for both 0 batch , 1 batch, and num tokens ensure that mask matches shape with attn scores ass context length may be 10,10 but input may be 2,6,3 so with num tokens it adjust mask to 6,6

        attn_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool,-torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1,2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        return(context_vec)
    

# Layer Normalization
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim=-1, keepdim = True, unbiased = False)
        norm_x = (x-mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    

# Feed forward Network with GELU Activation
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi))*(x + 0.044715 * torch.pow(x,3))
        ))
    
class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear( 4 * cfg["emb_dim"] , cfg["emb_dim"]),

        )

    def forward(self, x):
        return  self.layers(x)
    



# Actual Transformer block with all above features

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out= cfg["emb_dim"],
            context_length= cfg["context_length"],
            num_heads= cfg["n_heads"],
            dropout= cfg["drop_rate"],
            qkv_bias = cfg["qkv_bias"]
        )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        
        # layer 1 with shortcut connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # layer 2 with shortcut connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    

# GPT Configuration Dictionary
GPT_CONFIG_124M = {
    "vocab_size" :  50257,
    "context_length" : 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"],cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(torch.arange(seq_len,device=in_idx.device))
        x = tok_embeds+ pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
    


# batch
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim = 0)
print(batch)


# intializing model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)

print("Input batch: \n", batch)
print("output shape: \n", out.shape)
print("output batch: \n", out)


# Using numel() to find the total no. of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nThe total no of parameters: {total_params:,}")


print("\nToken embedding layer shape: ", model.tok_emb.weight.shape)
print("Output layer shape: ", model.out_head.weight.shape)

# claculating actual total no of parameters

total_params_gpt = total_params - sum(p.numel() for p in model.out_head.parameters() )
print(f"\nNo of trainale parameters considering weight tying: {total_params_gpt:,} ")

# compute the memory requirements of the 163 million parameters in our GPTModel object

total_size_bytes = total_params * 4
total_size_mb = total_size_bytes/ (1024 * 1024)
print(f"\nTotal size of the model: {total_size_mb:.2f} MB ")