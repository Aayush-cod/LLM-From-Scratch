import torch

vocab = {
    "closer" : 0,
    "every" : 1,
    "effort" : 2,
    "forward" : 3,
    "inches" : 4,
    "moves" : 5,
    "pizza" : 6,
    "toward" : 7,
    "you" : 8,
}

inverse_vocab = {v:k for k,v in vocab.items()}


# assume next token logits are for a context "every effort moves you"

next_token_logits = torch.tensor(
    [4.51, 0.89, -1.90, 6.75, 1.63, -1.62, -1.89, 6.28, 1.79]

)


# we can implement the top-k procedure

top_k = 3
top_logits , top_pos = torch.topk(next_token_logits, top_k)

print("Top logits: ", top_logits)
print("Top Positions: ", top_pos)

# we apply PyTorch's where function to set the logit values of tokens that are below the lowest logit value within our top-3 selection to negative infinity (-inf).

new_logits = torch.where(
    condition= next_token_logits < top_logits[-1],
    input = torch.tensor(float('-inf')),
    other = next_token_logits

)

print(new_logits)

# let's apply the softmax function to turn these into next-token probabilities:

topk_probas = torch.softmax(new_logits, dim = 0)
print(topk_probas)