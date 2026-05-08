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


# calculating probability distribution
probas = torch.softmax(next_token_logits, dim=0)

next_token_id = torch.argmax(probas).item()

print(inverse_vocab[next_token_id])


# To implement a probabilistic sampling process, we can now replace the argmax with the multinomial function in PyTorch:

torch.manual_seed(123)
next_token_id = torch.multinomial(probas, num_samples=1).item()
print(inverse_vocab[next_token_id])


def print_sampled_tokens(probas):
    torch.manual_seed(123)
    # sample is a list of tokenids for 1000 time [3,3,5,5,6,4,5,5,5........]
    sample = [torch.multinomial(probas, num_samples=1).item() for i in range(1_000)]
    # this count how much each token ids are there like forward is 555 times like this
    sampled_ids = torch.bincount(torch.tensor(sample))
    for i, freq in enumerate(sampled_ids):
        print(f"{freq} x {inverse_vocab[i]}")
    print("\n")

print_sampled_tokens(probas)


# using temperature scaling
def softmax_with_temperature(logits, temperature):
    scaled_logits = logits/temperature

    return torch.softmax(scaled_logits, dim=0)

scaled_probas_0 = softmax_with_temperature(next_token_logits, 0.1)
print_sampled_tokens(scaled_probas_0)

scaled_probas_5 = softmax_with_temperature(next_token_logits, 5)
print_sampled_tokens(scaled_probas_5)