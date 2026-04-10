import tiktoken

with open("Working_with_text_data/the-verdict.txt", "r", encoding="utf-8") as f:
    text = f.read()


tokenizer = tiktoken.get_encoding("gpt2")

enc_txt = tokenizer.encode(text)

print(len(enc_txt))

enc_sample = enc_txt[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]

print(f"x : {x}")
print(f"y:       {y}")


for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(f"{context}. --> {desired}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), " --> " ,tokenizer.decode([desired]))
