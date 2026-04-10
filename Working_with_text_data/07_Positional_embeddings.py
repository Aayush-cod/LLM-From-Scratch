import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self,txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []


        tokenids = tokenizer.encode(txt, allowed_special = {"<|endoftext|>"})

        for i  in range(0, len(tokenids)- max_length, stride):
            input_chunk = tokenids[i : i + max_length]
            target_chunk = tokenids[i+1 : i + max_length+1]

            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):
        return len(self.input_ids)
    

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader_v1(txt, batch_size = 4, max_length=256, stride = 128, shuffle = True, drop_last= True, num_workers =0):

    tokenizer = tiktoken.get_encoding("gpt2")

    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    dataloader = DataLoader(dataset, batch_size = batch_size,
                
                            shuffle = shuffle,
                            drop_last = drop_last, 
                            num_workers= 0
                            )
    
    return dataloader


with open("Working_with_text_data/the-verdict.txt", "r", encoding= "utf-8")as f:
    raw_txt = f.read()

max_length = 4

dataloader = create_dataloader_v1 ( 
    raw_txt, batch_size=8, max_length=max_length , stride= max_length, shuffle = False )



data_iter = iter(dataloader)
input, target = next(data_iter)

print("Input:\n", input)


vocab_size = 50257
# dimensionality of vector or inside a single token
output_dim = 256

# Conversion of Token Ids into vectors by embedding layer weight matrix look up method

token_embeddding_layer = torch.nn.Embedding(vocab_size, output_dim)

token_embedding = token_embeddding_layer(input)

print(token_embedding.shape)

# Positional embedding layer for each token embedding or vector
context_length = max_length
positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
positonal_embedding = positional_embedding_layer(torch.arange(context_length))

print(positonal_embedding.shape)


# Input embeddings to feed  LLMs

input_embeddings = token_embedding + positonal_embedding

print(input_embeddings.shape)





