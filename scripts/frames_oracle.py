import datasets

import sys
import urllib.parse
import urllib.request
from collections import defaultdict
import json 
import os 
sys.path.append('../')
from utils import get_oracle_ranking_filename
dataset_names = ['frames', 'frames_debug']
top_n_oracle_passages = 1
run_folder = '/nfs/data/calmar/amyrzakha/projects/runs'
split = 'test'

split_mapping = {'validation': 'dev', 'test': 'dev', 'train': 'train'}


for name in dataset_names:
    dataset = datasets.load_dataset("google/frames-benchmark", num_proc=10 )[split]
    out_file = get_oracle_ranking_filename(run_folder, f'{name}', split_mapping[split])
    with open(out_file, 'w') as fout:
        if 'debug' in out_file:
            dataset = dataset.select(range(min(len(dataset), 50)))
        for sample in dataset:                    
            query_id = str(sample['Unnamed: 0'])
            fout.write(f'{query_id}\tq0\t{query_id}\t{0}\t{top_n_oracle_passages-0}\trun\n')