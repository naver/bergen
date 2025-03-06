import numpy as np
import pandas as pd
import datasets
import torch
from tqdm import tqdm
import argparse
import os

from collections import Counter
import string
import re
import argparse
import json
import sys
import nltk
from vllm import LLM, SamplingParams
from torch.utils.data import DataLoader
from datasets import Dataset

# model
from vllm import LLM, SamplingParams
import torch
from transformers import AutoTokenizer

# dataset
# bergen/utils.py
import sys
sys.path.insert(0, "../..")
from utils import prepare_dataset_from_ids, load_trec
import datasets
import pickle

import argparse

parser = argparse.ArgumentParser(description='Silver labeling for Provence')
parser.add_argument('--model', type=str, required=False, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace model name")
parser.add_argument('--datastore', required=False, default="../../datasets/ms-marco-docs-v1-chunked-v1_full", help="path to a dataset saved by bergen")
parser.add_argument('--queries', required=False, default="../../datasets/ms-marco-docs-v1-queries-dev_full", help="path to a dataset saved by bergen") 
parser.add_argument('--trec', required=False, default="../../runs/run.rerank.retriever.top_50.naver_splade-v3.rerank.top_50.ms-marco-docs-v1-queries-dev.ms-marco-docs-v1-chunked-v1.dev.naver_trecdl22-crossencoder-debertav3.trec", help="trec run (reranking for the provided queries+datastore) saved by bergen")
parser.add_argument('--top_k', type=int, default=5, help="how many top retrieved+reranked documents to label")
parser.add_argument('--batch_size', type=int, default=64, help="how many top retrieved+reranked documents to label")
parser.add_argument('--prompt', type=str, required=False, choices=["answer", "relevant", "straightforward"], default="answer", help="which prompt to use for data labeling")
parser.add_argument('--outdir', type=str, required=True, help="a new folder name where to save all the generated labeling, one example per file")
parser.add_argument('--overwrite', action='store_true', help="overwrite the existing labeling")
parser.add_argument('--max_new_tokens', type=int, default=256, 
                    help="max new tokens to be generated")
parser.add_argument('--skip', type=int, default=1, 
                   help="step in enumerating queries in the dataset, can be used for debug e.g. --skip=1000")

args = parser.parse_args()

# set up output folder
if os.path.exists(args.outdir) and not args.overwrite:
    raise ValueError(f"Path {args.outdir} exists! If you want to overwrite it set --overwrite flag, otherwise provide a new unique outdir")
os.makedirs(args.outdir, exist_ok=True)

# labeler LLM
# the code was run with "meta-llama/Meta-Llama-3-8B-Instruct"
# in case of using another model check if it works well with the reference format used throughout the code, i.e. [i]
# e.g. Command-R was trained with another reference format <co: i></co: i>

tokenizer = AutoTokenizer.from_pretrained(args.model)
model = LLM(model=args.model, tensor_parallel_size=torch.cuda.device_count(), dtype=torch.float16, 
            gpu_memory_utilization=0.9, max_model_len=2048, enforce_eager=True, kv_cache_dtype="fp8_e5m2")
sampling_params = SamplingParams(
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,  # Temperature set to 0 for greedy decoding
                )

# dataset
train = datasets.load_from_disk(args.queries)
id2index = pickle.load(open(args.queries+'/id2index.p', 'rb'))
train.id2index = id2index

wiki = datasets.load_from_disk(args.datastore)
id2index = pickle.load(open(args.datastore+'/id2index.p', 'rb'))
wiki.id2index = id2index

query_ids, doc_ids, scores = load_trec(args.trec)

doc_ids2 = [[d] for idx, (q, doc_ids_q) in enumerate(zip(query_ids, doc_ids)) for d in doc_ids_q[:args.top_k]]
query_ids = [q for idx, (q, doc_ids_q) in enumerate(zip(query_ids, doc_ids)) for d in doc_ids_q[:args.top_k]]
doc_ids = doc_ids2

processed_datasets = {
        "query": train,
        "doc": wiki,
    }

gen_dataset = prepare_dataset_from_ids(
            processed_datasets, 
            query_ids, 
            doc_ids,
            multi_doc=False,
            )

# prompt
if args.prompt == "answer":
    prompt_template = "Question: %s\n\nContext:\n%s\n\nAnswer the Question, using ONLY information provided in the Context. If no useful information is provided, you MUST output “No answer”. If some parts of the Context are used to answer, you MUST cite ALL the corresponding sentences. Use the symbols [ ] to indicate when a fact comes from a sentence in the context, e.g [0] for a fact from sentence 0. You should only answer the given question and should not provide any additional information. "
elif args.prompt == "relevant":
    prompt_template = "Question: %s\n\nContext:\n%s\n\nSummarize which information, generally relevant to the given Question, the given Context provides. If no useful information is provided, you MUST output “No answer”. If some parts of the Context are relevant (fully or partially), you MUST summarize it and you MUST cite ALL the corresponding sentences. Use the symbols [ ] to indicate when a fact comes from a sentence in the context, e.g [0] for a fact from sentence 0. "
elif args.prompt == "straightforward":
    prompt_template = "Question: %s\n\nContext:\n%s\n\nOutput the indexes of the sentences which contain an answer to the given Question. Use the symbols [ ] to select sentences, e.g. [0, 5] for selecting sentences 0 and 5. If no useful information is provided, you MUST output “No answer”. "
else:
    raise ValueError(f"Prompt {args.prompt} is not supported")

# prompts and dataloader
prompts, sents_all = [], []
for q, cntx in zip(gen_dataset["query"][::args.skip], gen_dataset['doc'][::args.skip]):
    if cntx[0] == ".": cntx = cntx[1:].strip()
    sents = nltk.sent_tokenize(cntx)
    sents_all.append(sents)
    cntx = " ".join([f"[{i}] {s}" for i, s in enumerate(nltk.sent_tokenize(cntx))])
    prompt = prompt_template % (q, cntx)
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(messages, 
                                           tokenize=False, 
                                           add_generation_prompt=True)
    prompts.append(prompt)
    
dataset = Dataset.from_dict({"prompts": prompts, 
                             "sents": sents_all, 
                             "queries": gen_dataset["query"][::args.skip], 
                             "id": [qid+"_"+did for qid, did in zip(gen_dataset["q_id"], gen_dataset["d_id"])][::args.skip]})

def mycollate(batch):
    return {key:[item[key] for item in batch] for key in batch[0]}
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=mycollate)

for batch in tqdm(dataloader, desc='Pruning contexts...', total=len(dataloader)):
    outputs = model.generate(batch["prompts"], sampling_params)
    for (sents, query, id_, output) in zip(batch["sents"], batch["queries"], batch["id"], outputs):
        response = output.outputs[0].text
        if "No answer" in response:
            selected_sents = []
        else:
            matches = re.findall(r'\[([\d, ]+)\]', response)
            try:
                selected_idxs = set([int(num) for match in matches for num in match.split(',')])
            except:
                continue
            if len(selected_idxs) == 0:
                # simple filter: if "No answer" is not generated, there should be at least one selected sentence
                continue
            selected_sents = [i for i, sent in enumerate(sents) if i in selected_idxs]
        with open(os.path.join(args.outdir, id_+".json"), "w") as fout:
            res_item = {
                "query": query,
                "context": sents,
                "selected_sents": selected_sents,
                "response": response,
            }
            fout.write(json.dumps(res_item))