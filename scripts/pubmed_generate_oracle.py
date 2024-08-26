import datasets

import sys
import urllib.request
from collections import defaultdict
import json 
import os 
sys.path.append('../')
from utils import get_oracle_ranking_filename
dataset_names = ['bioasq11b']
top_n_oracle_passages = 100
run_folder = '../runs'
split = 'train'

split_mapping = {'validation': 'dev', 'test': 'dev', 'train': 'dev'}



for name in dataset_names:
    path='/beegfs/scratch/project/calmar/data/bioasq11btrain/bioasq11btrain.tsv'
    #dataset = datasets.load_dataset('kilt_tasks', name )[split]
    dataset = datasets.load_dataset("csv", data_files=[path], delimiter="\t", )[split]
    out_file = get_oracle_ranking_filename(run_folder, f'{name}', split_mapping[split])
    print (out_file)
    with open(out_file, 'w') as fout:
        for sample in dataset:
            query_id = str(sample['id'])
    
            # get passage_ids
            passage_ids = eval(sample['docs'])
            #passage_ids = passage_ids[:top_n_oracle_passages]


            for i, passage_id in enumerate(passage_ids):
                fout.write(f'{query_id}\tq0\t{passage_id}\t{i}\t{top_n_oracle_passages-i}\trun\n')
                
