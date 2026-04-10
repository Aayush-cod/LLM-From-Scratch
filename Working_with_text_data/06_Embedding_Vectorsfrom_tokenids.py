import torch

input_ids = torch.tensor([2,3,5,1])

vocab_size = 6


# Dimenson size of vector in each sample like [0.15 , -0.234, -0.95]
output_dim = 3

torch.manual_seed(123)

embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
print(embedding_layer.weight)

# embedding vector check for a token id 3

print(embedding_layer(torch.tensor([3])))

# embedding vector check for all token ids or input_ids

print(embedding_layer(input_ids))