import torch

inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your (x^1)
[0.55, 0.87, 0.66], # journey (x^2)
[0.57, 0.85, 0.64], # starts (x^3)
[0.22, 0.58, 0.33], # with (x^4)
[0.77, 0.25, 0.10], # one (x^5)
[0.05, 0.80, 0.55]] # step (x^6)
)

x_2 = inputs[1]

d_in = inputs.shape[1]
print(d_in)
d_out = 2

# initialize the three weight matrices Wq, Wk, and Wv

torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad = False)
W_key = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in,d_out), requires_grad = False)


# compute the query, key, and value vectors for query i.e 2

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value
print(query_2)

# key and value vectors for all input elements

keys = inputs @ W_key
values = inputs @ W_value
print("\n Keys.shape: ", keys.shape)
print("\n Values.shape: ", values.shape)


# Calculate attention score of query 2 key

keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)


# Calculate attention score of all inputs based on query 2 key
# All attention scores for given query

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

#  we compute the attention weights by scaling the
# attention scores and using the softmax function we used earlier. The difference to earlier is
# that we now scale the attention scores by dividing them by the square root of the
# embedding dimension of the keys,

d_k = keys.shape[-1]

attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim = -1)
print(attn_weights_2)

# we now compute the context vector as a weighted sum over the value
# vectors. 

context_vec_2 = attn_weights_2 @ values
print(context_vec_2)