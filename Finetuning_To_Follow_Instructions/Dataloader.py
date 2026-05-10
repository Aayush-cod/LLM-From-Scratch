import torch
from  Finetuning_To_Follow_Instructions.Dataset_Collat_Function import InstructionDataset
from Finetuning_To_Follow_Instructions.Dataset_Collat_Function import custom_collate_fn
from Finetuning_To_Follow_Instructions.Dataset_Collat_Function import tokenizer

from Finetuning_To_Follow_Instructions.Dataset_preparation import train_data
from Finetuning_To_Follow_Instructions.Dataset_preparation import val_data
from Finetuning_To_Follow_Instructions.Dataset_preparation import test_data


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    # Use PyTorch 2.9 or newer for stable mps results
    major, minor = map(int, torch.__version__.split(".")[:2])
    if (major, minor) >= (2, 9):
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
else:
    device = torch.device("cpu")




# we use the partial function from Python's functools standard library to create a new version of the function with the device argument pre-filled.

from functools import partial

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# we instantiate the data loaders similar to previous chapters, except that we now provide our own collate function for the batching process

from torch.utils.data import DataLoader


num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)

val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)


if __name__ == "__main__":
    print("Device:", device)

    print("Train loader:")
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)

    # checks for input with end of text token id = 50256
    print("\n",inputs[0])

    # checks targte with -100 which replaced of 50256
    print("\n",targets[0])