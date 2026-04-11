import torch

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)


# Calculate attention scores with respect to query using dot product
query = inputs[1]

att_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    att_scores_2[i] = torch.dot(x_i, query)

print("Attention_scores: ", att_scores_2)

# Normalizing attention score to calculate attention weight
# Step1 - normal method

attn_weights_2_tmp = att_scores_2/ att_scores_2.sum()

print("Attn_weight", attn_weights_2_tmp)
print("Attention_weight_sum: ", attn_weights_2_tmp.sum())


# Step2- softmax_naive method

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim= 0)

attn_weights_2_naive = softmax_naive(att_scores_2)
print("\nAttn_weight_softmax\n", attn_weights_2_naive)
print("\nAttention_weight_softmax_sum: \n", attn_weights_2_naive.sum())

# Pytorch implementation of softmax

attn_weights_2 = torch.softmax(att_scores_2, dim = 0)
print("\nAttn_weight: ", attn_weights_2_tmp)
print("\nAttention_weight_sum: ", attn_weights_2_tmp.sum())



