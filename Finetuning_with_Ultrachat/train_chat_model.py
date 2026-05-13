import torch
from torch.utils.data import DataLoader
import tiktoken

from Transformer_Blocks_Implementing_GPT_Model.GPT_model_To_Generate_Text_08 import GPTModel
from Finetuning_with_Ultrachat.config import BASE_CONFIG
from Finetuning_with_Ultrachat.chat_dataset import ChatDataset
from Finetuning_with_Ultrachat.chat_collate import chat_collate


if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

tokenizer = tiktoken.get_encoding("gpt2")

# ✅ Load instruction-finetuned model
model = GPTModel(BASE_CONFIG)
model.load_state_dict(
    torch.load("gpt2-medium355M-sft.pth", map_location=device)
)
model.to(device)
model.train()

# ✅ Load dataset
dataset = ChatDataset("chat_sft_dataset.json", tokenizer)
loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=chat_collate)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

num_epochs = 2
from tqdm import tqdm

for epoch in range(num_epochs):
    total_loss = 0

    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch+1}"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-100
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # This helps prevent MPS memory buildup.
        del logits, loss
        torch.mps.empty_cache()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# ✅ Save new chat-finetuned model
torch.save(model.state_dict(), "gpt2-medium355M-chat3-sft.pth")
print("✅ Chat model saved as gpt2-medium355M-chat-sft.pth")