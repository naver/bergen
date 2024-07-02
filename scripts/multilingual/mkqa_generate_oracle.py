import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(dir_path+'/../..')

import datasets
import urllib.request
from collections import defaultdict
import json 
from utils import get_oracle_ranking_filename

lng = sys.argv[1]
lng_mkqa = lng if lng != "zh" else "zh_cn"
name = f"mkqa_{lng}" # dataset name from module/dataset_processor.py

top_n_oracle_passages = 100
run_folder = dir_path + '/../../runs'
out_file = get_oracle_ranking_filename(run_folder, name, "dev")

# create dataset
mkqa = datasets.load_dataset('mkqa')
kilt_nq = datasets.load_dataset("kilt_tasks", "nq")

mkqa_ids = {s['example_id']:i for i, s in enumerate(mkqa["train"])}
kilt_nq_train_ids = {s['id']:i for i, s in enumerate(kilt_nq["train"])}

overlap_ids = set(mkqa_ids.keys()).intersection(set(kilt_nq_train_ids.keys()))
overlap_mkqa = mkqa['train'].select([mkqa_ids[i] for i in overlap_ids])
overlap_kilt_nq = kilt_nq['train'].select([kilt_nq_train_ids[i] for i in overlap_ids])        
dataset = overlap_kilt_nq.add_column(f"content", [sample['queries'][lng_mkqa] for sample in overlap_mkqa])    
# discarding empty answers
dataset = dataset.add_column(f"label", [[a['text'] for a in sample['answers'][lng_mkqa] if not a['text']==None] for sample in overlap_mkqa])
# filter out samples with empty answer
dataset = dataset.filter(lambda example: len(example['label'])>0)

# ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el[f'answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})        
dataset = dataset.remove_columns(['meta'])

# write oralce
with open(out_file, 'w') as fout:
    for sample in dataset:
        query_id = sample['id']
        paragraph_bounds, wiki_ids = list(), list()
        for out in sample['output']:
            if len(out['answer']) > 0 and 'provenance' in out and out['provenance'] != None:
                bounds_out = [(prov['start_paragraph_id'], prov['end_paragraph_id']) for prov in out['provenance']]
                wiki_id_out = [prov['wikipedia_id'] for prov in out['provenance']]
                if len(bounds_out) > 0 and len(wiki_id_out) > 0:
                    paragraph_bounds.append(bounds_out)
                    wiki_ids.append(wiki_id_out)
                    
        # get passage_ids
        passage_ids = list()
        for id_, bounds_out, wiki_id_out in zip(query_id, paragraph_bounds, wiki_ids):
            for bounds, wiki_id in zip(bounds_out, wiki_id_out):
                start, end = bounds
                for i in list(range(start,end+1)):
                    pass_id = f'{wiki_id}_{i+1}'
                    passage_ids.append(pass_id)
        
        passage_ids = list(set(passage_ids))
        passage_ids = passage_ids[:top_n_oracle_passages]


        for i, passage_id in enumerate(passage_ids):
            fout.write(f'{query_id}\tq0\t{passage_id}\t{i}\t{top_n_oracle_passages-i}\trun\n')
                
