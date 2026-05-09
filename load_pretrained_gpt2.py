from gpt_download import download_and_load_gpt2

# Download and load GPT-2 124M


settings, params = download_and_load_gpt2(
        model_size="124M",
        models_dir="gpt2"
    )

if __name__ == "__main__":
    print("Settings:", settings)
    print("Parameter dictionary keys:", params.keys())
    print("Token embedding weight shape:", params["wte"].shape)