import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self,txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []


        tokenids = tokenizer.encode(txt, allowed_special = {"<|endoftext|>"})

        # range(start, stop, step), step → how much to jump each time, Why len(tokenids) - max_length?
        # Because inside the loop we do:

        # Python

        # input_chunk = tokenids[i : i + max_length]
        # If i becomes too large, i + max_length would go past the list.

        # So we stop early to avoid overflow.


        

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


dataloader = create_dataloader_v1 ( 
    raw_txt, batch_size=1, max_length=4 , stride= 1, shuffle = False )

data_iter = iter(dataloader)

first_batch = next(data_iter)

# First batch of input target pairs
print("Firstbatch:\n",first_batch)


second_batch = next(data_iter)

print("secondbatch:\n",second_batch)

dataloader = create_dataloader_v1 ( 
    raw_txt, batch_size=8, max_length=4 , stride= 4, shuffle = False )

data_iter = iter(dataloader)
input, target = next(data_iter)

# 8 batch of input and tragets

print("Input:\n", input)
print("target:\n", target)



