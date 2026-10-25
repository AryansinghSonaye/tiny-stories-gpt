import numpy as np
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset , DataLoader
from model import *

GPT_CONFIG_124M = {
    "vocab_size" : 50257 , # Vocabulary size
    "context_length" : 256 , # Context length
    "emb_dim" : 768 , # Embedding dimension
    "n_heads" : 12 , # Number of attention heads
    "n_layers" : 12 , # Number of layers
    "drop_rate" : 0.1 , # Dropout rate
    "qkv_bias" : False   # Query-Key-Value bias
}

def generate_text_simple(model , idx , max_new_tokens , context_size) :
    for _ in range(max_new_tokens) :
        idx_cond = idx[: , -context_size:]

        with torch.no_grad() :
            logits = model(idx_cond)
        logits = logits[:,-1,:]
        probas = torch.softmax(logits , dim=-1)
        idx_next = torch.argmax(probas , dim=-1 , keepdim=True)
        idx = torch.cat((idx , idx_next) , dim=1)

    return idx


def text_to_token_ids(text , tokenizer) :
    encoded = tokenizer.encode(text , allowed_special={"<|endoftext|>"})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids , tokenizer) :
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def generate(model , idx , max_new_tokens , context_size , temperature=0.0 , top_k=None , eos_id=None) :
    for _ in range(max_new_tokens) :
        idx_cond = idx[: , -context_size:]

        with torch.no_grad() :
            logits = model(idx_cond)

        logits = logits[: , -1 , :]

        if top_k is not None :
            top_logits , top_pos = torch.topk(logits , top_k)
            new_logits = torch.where(condition = logits < top_logits[: , -1] , input = torch.tensor(float("-inf")) , other = logits)
            logits = new_logits

        if temperature > 0.0 :
            logits = logits / temperature
            probs = torch.softmax(logits , dim=-1)
            idx_next = torch.multinomial(probs , num_samples=1)

        else :
            idx_next = torch.argmax(logits , dim=-1 , keepdim=True) # (batch , 1)

        if eos_id is not None and idx_next.item() == eos_id:
            break

        idx = torch.cat((idx , idx_next) , dim=1) # (batch , n_tokens+1)

    return idx

tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPTModel(GPT_CONFIG_124M)

model.load_state_dict(
    torch.load("TinyStoryPunyGPT.pth", map_location=device)
)

model.to(device)
model.eval()

while True:
    prompt = input("Prompt: ")

    if prompt.lower() == "exit":
        break

    token_ids = text_to_token_ids(prompt , tokenizer).to(device)

    out = generate(model , token_ids , max_new_tokens=100 , context_size=GPT_CONFIG_124M["context_length"] , temperature=0.8 , top_k=40)

    print(token_ids_to_text(out , tokenizer))

