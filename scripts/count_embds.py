import torch
import os
from tqdm import tqdm

folder = 'indexes/kilt-100w_doc_naver_splade-cocondenser-selfdistil/'
folder = 'indexes/kilt-100w_doc_castorini_repllama-v1-7b-lora-passage/'
count = 0
for f in tqdm(os.listdir(folder)):
    count += torch.load(folder + f).shape[0]
print(count)
