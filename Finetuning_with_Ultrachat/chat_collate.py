import torch
import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

assistant_token_ids = tokenizer.encode("Assistant:")
user_token_ids = tokenizer.encode("User:")

def chat_collate(batch, pad_token_id=50256):

    max_length = max(len(item) for item in batch) + 1

    inputs_list = []
    targets_list = []

    for item in batch:

        tokens = item.tolist() + [pad_token_id]
        padded = tokens + [pad_token_id] * (max_length - len(tokens))

        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        # Mask padding
        targets[targets == pad_token_id] = -100

        # Start with everything masked
        mask = torch.ones_like(targets) * -100

        i = 0
        while i < len(tokens):

            # Detect Assistant:
            if tokens[i:i+len(assistant_token_ids)] == assistant_token_ids:

                i += len(assistant_token_ids)

                # Unmask assistant reply tokens
                while i < len(tokens):

                    # Stop at next User:
                    if tokens[i:i+len(user_token_ids)] == user_token_ids:
                        break

                    # Align with targets (shifted by 1)
                    if i-1 < len(mask):
                        mask[i-1] = targets[i-1]

                    i += 1
            else:
                i += 1

        targets = mask

        inputs_list.append(inputs)
        targets_list.append(targets)

    return torch.stack(inputs_list), torch.stack(targets_list)