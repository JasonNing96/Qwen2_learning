from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm
from typing import List

from models.model import ModelArgs, Transformer,LLaMA


def inference():
    torch.manual_seed(0)

    allow_cuda = True
    device = 'cuda:0' 
    # device = 'cpu' 
    # if torch.cuda.is_available() and allow_cuda else 'cpu'

    prompts = [
        "Simply put, the theory of relativity states that ",
        "If Google was an Italian company founded in Milan, it would"
    ]

    # prompts = [""" Write a cute happy ending story for this :
    #
    #            Revathi and abishek met in bumble they were in long distance relationship for 5 months and decided to marry each other """]

    model = LLaMA.build(
        checkpoints_dir='/home/njh/LLM/TinyLlama-1.1B-Chat-v1.0/',
        tokenizer_path='/home/njh/LLM/TinyLlama-1.1B-Chat-v1.0/tokenizer.model',
        # tokenizer_path='tokenizer.model',
        load_model=True,
        max_seq_len=512,
        max_batch_size=len(prompts),
        device=device
    )
    print("Model loaded successfully.")
    # while True:
    #     print('Welcome to Abi LLama 2 ')
    #     input_text = input('Enter the promt (exit to stop): ')
    #     prompts=[input_text]
    out_tokens, out_texts = (model.text_completion(prompts, max_gen_len=100))
    assert len(out_texts) == len(prompts)
    for i in range(len(out_texts)):
        print(f'{out_texts[i]}')
        print('-' * 50)
    

if __name__ == '__main__':
    inference()