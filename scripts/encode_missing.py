#from models.retrievers.splade import Splade
from models.retrievers.repllama import RepLlama
import datasets
import torch
from tqdm import tqdm

batch_size = 128
dataset = datasets.load_from_disk('datasets/kilt-100w_full')
content = dataset['content'][24731520:]
#retriever = Splade('naver/splade-cocondenser-selfdistil')
retriever = RepLlama('castorini/repllama-v1-7b-lora-passage')
print(len(content))
embs = list()
batch_size=512
with torch.no_grad():
    print(len(content))
    for i in tqdm(range(0, len(content), batch_size)):
        batch = content[i:i+batch_size]
        print(len(batch))
        inp = retriever.tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
        out = retriever(inp)['embedding'].cpu().detach().to_sparse()
        embs.append(out)
embs = torch.cat(embs)
torch.save(embs, 'repllama_missing.pt')


