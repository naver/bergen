import datasets

import sys
import urllib.request
from collections import defaultdict
import json 
import os 
sys.path.append('../')
from utils import get_oracle_ranking_filename

top_n_oracle_passages = 100
run_folder = '../runs'
split = 'train'

split_mapping = {'validation': 'dev', 'test': 'test', 'train': 'train'}

dataset = datasets.load_dataset("jenhsia/ragged", 'bioasq')["train"]
# ['id', 'input', 'output', 'question_type']

out_file = get_oracle_ranking_filename(run_folder, 'BIOASQ11B_Ragged', split_mapping[split])
with open(out_file, 'w') as fout:
    for sample in dataset:
        query_id = sample['id']
        paragraph_bounds, pubmed_ids = list(), list()

        for out in sample['output']:
            if 'provenance' in out and out['provenance'] is not None:
                assert len(out['provenance']) == 1
                pubmed_id = out['provenance'][0]['page_id']
                bounds_out = [(prov['start_par_id'], prov['end_par_id']) for prov in out['provenance']]
                pubmed_id_out = [prov['page_id'] for prov in out['provenance']]
                if len(bounds_out) > 0 and len(pubmed_id_out) > 0:
                    paragraph_bounds.append(bounds_out)
                    pubmed_ids.append(pubmed_id_out)
        
        # get passage_ids
        passage_ids = list()
        for id_, bounds_out, pubmed_id_out in zip(query_id, paragraph_bounds, pubmed_ids):
            for bounds, pubmed_id in zip(bounds_out, pubmed_id_out):
                start, end = bounds
                for i in list(range(start,end+1)):
                    pass_id = f'{pubmed_id}'
                    passage_ids.append(pass_id)
    
        passage_ids = list(set(passage_ids))
        passage_ids = passage_ids[:top_n_oracle_passages]

        for i, passage_id in enumerate(passage_ids):
            fout.write(f'{query_id}\tq0\t{passage_id}\t{i}\t{top_n_oracle_passages-i}\trun\n')
            
