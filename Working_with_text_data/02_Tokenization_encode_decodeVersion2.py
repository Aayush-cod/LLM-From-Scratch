
import re

text = """ "Hello world, I am a test. """

split1 = re.split(r'([",.]|\s)', text)
print(split1)

split2 = [item for item in split1 if item.strip()]
print(split2)

vocabulary = sorted(list(set(split2)))
print(vocabulary)

# endoftext seperates unknown two seperate files and unk denotes unkown words not from vocabulary

vocabulary.extend(["<|endoftext|>","<|unk|>"])

vocab = {token:index for index , token in enumerate(vocabulary)}
print(vocab)

# for i, item in enumerate(vocab.items()):
#     print(item)
#     if i >= 4:
#         break

for i, item in enumerate(list(vocab.items())[-3:]):
    print(item)
    
class TokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {index:token for token, index in vocab.items()}
        
    def encode(self, text):
        process = re.split(r'([",.]|\s)', text)
        process = [item for item in process if item.strip()]
        process = [item if item in self.str_to_int else "<|unk|>" for item in process]
        ids = [self.str_to_int[token] for token in process]
        return ids
        
    def decode(self, ids):
        text = " ".join([self.int_to_str[index] for index in ids])
        text = re.sub(r'\s+([",.])',r'\1', text)
        return text
        
        
tokenizer = TokenizerV2(vocab)

text1 = " Hey Hello world, Test."

text2 = "I am world king."

texts = " <|endoftext|> ".join((text1,text2))

print(texts)

encoded = tokenizer.encode(texts)
print("Encoded: ", encoded)

decoded = tokenizer.decode(encoded)
print("Deocded: ", decoded)
        








