import datasets

import sys
import urllib.parse
import urllib.request
from collections import defaultdict
import json 
import os 
sys.path.append('../')
from utils import get_oracle_ranking_filename
dataset_names = ['TimeSensitiveQA']
top_n_oracle_passages = 100
run_folder = '../runs'
split = 'validation'

split_mapping = {'validation': 'dev', 'test': 'dev', 'train': 'train'}



for name in dataset_names:
    dataset = datasets.load_dataset("diwank/time-sensitive-qa", num_proc=10 )[split]
    out_file = get_oracle_ranking_filename(run_folder, f'{name}', split_mapping[split])
    print (out_file)
    with open(out_file, 'w') as fout:
        for sample in dataset:
            query_id = str(sample['idx'])
    
            # get url
            passage_ids = [ "https://en.wikipedia.org"+urllib.parse.quote (f'{sample["idx"].split("#")[0]}').replace('_','%20') ]

            for i, passage_id in enumerate(passage_ids):
                fout.write(f'{query_id}\tq0\t{passage_id}\t{i}\t{top_n_oracle_passages-i}\trun\n')
                
