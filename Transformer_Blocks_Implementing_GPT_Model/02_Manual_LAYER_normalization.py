import torch
import torch.nn as nn


torch.manual_seed(123)
batch_example = torch.randn(2,5)
# Linear layer we studied in previous multi head attention to use insetad of parameter for creating weight parameters
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)
print(out)

# Output layer mean and var checking 
mean = out.mean(dim= -1, keepdim = True)
var = out.var(dim= -1 , keepdim = True)
print("mean: \n", mean)
print("var: \n", var)


# Layer Normalization

out_norm = (out - mean)/ torch.sqrt(var)
mean = out_norm.mean(dim= -1, keepdim = True)
var = out_norm.var(dim= -1 , keepdim = True)
print("Normalized layer outputs: \n", out_norm)
print("mean: \n", mean)
print("var: \n", var)

# for mean = 0, we can also turn off the scientific notation when printing tensor values by setting sci_mode to False:
torch.set_printoptions(sci_mode=False)
print("mean: \n", mean)
print("var: \n", var)

