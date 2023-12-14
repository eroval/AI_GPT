#Dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# review some of the data - including number of characters in text
print("\n[STEP] #01:")
print("length of dataset in characters: ", len(text))
print(text[:1000])

print("\n[STEP] #02:")
# store unique characters from out text
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

print("\n[STEP] #03:")
# create a mapping from characters to integers 
# there are also more complex techniques than 1:1 mapping of letters
# such as tokenizers - it could be words or subwords (tiktoken, google)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# encode the entire text dataset and store it into a torch.Tensor
import torch # we use PyTorch: https://pytorch.org
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000]) # the 1000 characters we looked at earier will to the GPT look like this