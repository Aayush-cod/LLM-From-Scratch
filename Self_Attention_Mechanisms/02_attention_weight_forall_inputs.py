import torch

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

# For loop method - slow

attn_scores = torch.empty(6,6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i,x_j)

print("\n Attention Scores: ", attn_scores)


# Matrix multiplication method - fast

attn_scores = inputs @ inputs.T
print("\n Attention_Scores 02 : ", attn_scores)

#Attention weight

attn_weights = torch.softmax(attn_scores, dim = -1)
print("\n Attn-weights: ",attn_weights)

# Check row sums up to 1

row_2 = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("\n row_2 sum : ", row_2)

print("\n All inputs sum : ", attn_weights.sum(dim=-1))


# Context vectors using matrix multiplication

cntx_vectors = attn_weights @ inputs
print("\n Context_vectors: ", cntx_vectors)