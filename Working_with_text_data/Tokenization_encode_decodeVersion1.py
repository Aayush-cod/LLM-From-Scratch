import re

text = "Hello, world. This, is a test."

# Convert text into token individual text +special characters

split1 = re.split(r'([,.]|\s)', text)
print(split1)

processed = [item for item in split1 if item.strip()]
processed_len = len(processed)
print(processed_len)
print(processed)

# Create Vocabulary i.e arrange tokens in alphabetical order and remove duplicates and map string to integer i.e token ids

vocabulary = sorted(set(processed))
print(vocabulary)

# Mapping string to integer i.e assining token ID's

vocab = { token:index for index, token in enumerate(vocabulary)}

# Dictionary : Vocab
print(vocab)

# For larger dictionaries

for i , item in enumerate(vocab.items()):
    print(item)
    if i >= 4:
        break

#Create a class tokenizer with encode and decode function

class TokenizerV1:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {index:token for token,index in vocab.items()} #Reverse int to string i.e decode

    def encode(self, text):
        process = re.split(r'([,.]|\s)',text)
        process = [item for item in process if item.strip()]
        ids = [self.str_to_int[token] for token in process]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[index] for index in ids])
        text = re.sub(r'\s + ([,.])', r'\1',text)
        return text
    

tokenizer = TokenizerV1(vocab)
    

texts= "Hello, test."

encoded = tokenizer.encode(texts)
print("Encoded: ", encoded )

decoded = tokenizer.decode(encoded)
print("Decoded: ", decoded)