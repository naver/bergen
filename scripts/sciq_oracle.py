import datasets

import sys
import urllib.parse
import urllib.request
from collections import defaultdict
import json 
import os 
sys.path.append('../')
from utils import get_oracle_ranking_filename
dataset_names = ['sciq']
top_n_oracle_passages = 100
run_folder = '../runs'
split = 'test'

split_mapping = {'validation': 'dev', 'test': 'dev', 'train': 'train'}



for name in dataset_names:
    dataset = datasets.load_dataset("sciq", num_proc=10 )[split]
    out_file = get_oracle_ranking_filename(run_folder, f'{name}', split_mapping[split])
    print (out_file)
    cid= [split+str(i) for i in range(len(dataset))]
    dataset = dataset.add_column("id", cid)
    with open(out_file, 'w') as fout:
        for sample in dataset:
            query_id = str(sample['id'])
            fout.write(f'{query_id}\tq0\t{query_id}\t{0}\t{top_n_oracle_passages-0}\trun\n')
                
