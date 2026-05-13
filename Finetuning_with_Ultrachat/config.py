"""
Configuration file for GPT‑2 Medium (355M)
Used for loading the finetuned chatbot model.
"""

BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "drop_rate": 0.0,        # No dropout during inference
    "qkv_bias": True,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16
}