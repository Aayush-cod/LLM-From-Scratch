import torch
import tensorflow as tf
import tqdm
import tiktoken

print("Torch MPS:", torch.backends.mps.is_available())
print("TF version:", tf.__version__)
print("All good ✅")