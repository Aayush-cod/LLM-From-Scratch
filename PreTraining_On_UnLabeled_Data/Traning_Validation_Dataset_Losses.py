import torch
import tiktoken 

tokenizer = tiktoken.get_encoding("gpt2")
from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel

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

with open("PreTraining_On_UnLabeled_Data/the-verdict.txt", "r", encoding= "utf-8")as f:
    text_data = f.read()


total_characters = len(text_data)
totel_tokens = len(tokenizer.encode(text_data))
print("Characters: ", total_characters)
print("tokens: ", totel_tokens)

# Splitting into 90% tranning and 10% validation dataset

train_ratio = 0.9
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

# create the respective data loader reusing the create_dataloader_v1 code from chapter 2
from Working_with_text_data.Dataset_Dataloader_pytorch_05 import create_dataloader_v1

train_loader = create_dataloader_v1(
    train_data,
    batch_size= 2,
    max_length= GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size= 2,
    max_length= GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

# val = iter(val_loader)
# val_= next(val)
# print("\nval loader: ",val_)

print("\nTrain loader: ")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\nVal Loader: ")
for x, y in val_loader:
    print(x.shape, y.shape)


# implement a utility function to calculate the cross entropy loss of a given batch returned via the training and validation loader:

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(logits.flatten(0,1),
                                             target_batch.flatten())
    return loss

# implement the following calc_loss_loader function that computes the loss over all the batches sampled by a given data loader:

def calc_loss_loader(dataloader, model, device, num_batches = None):
    total_loss = 0
    if len(dataloader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(dataloader)
    else:
        num_batches = min(num_batches, len(dataloader))
    
    for i , (input_batch, target_batch) in enumerate(dataloader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch , model, device)
            total_loss += loss.item()

        else:
            break
    return total_loss/ num_batches


# intialize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


if __name__ == "__main__":

        with torch.no_grad():
            train_loss = calc_loss_loader(train_loader,model, device )
            val_loss = calc_loss_loader(val_loader,model, device )

            print("\nTraining Loss: ", train_loss)
            print("\nValidation Loss: ", val_loss)