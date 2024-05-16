import datasets

import sys
import urllib.request
from collections import defaultdict
import json 
import os 
sys.path.append('../')
from utils import get_oracle_ranking_filename
dataset_names = ['nq']
#dataset_names = ['aidayago2', 'cweb', 'eli5', 'fever', 'hotpotqa', 'nq', 'structured_zeroshot', 'trex', 'triviaqa_support_only', 'wned', 'wow']
top_n_oracle_passages = 100
run_folder = '../runs'
split = 'train'

split_mapping = {'validation': 'dev', 'test': 'test', 'train': 'train'}


def add_data_eli5(dataset):
    os.makedirs('../data', exist_ok=True)
    original_fname = '../data/eli5-dev-kilt.jsonl'
    if not os.path.exists(original_fname):
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl", original_fname)

    original = defaultdict(dict)
    for l in open(original_fname):
        example = json.loads(l)
        original_provenance = [out['provenance'] for out in example['output'] if 'provenance' in out]
        q_id = example['id']
        original[q_id] = original_provenance

    def add_missing_data(x, original_dataset):
        id_ = x['id']
        original_provenance = original_dataset[id_]
        for i in range(len(original_provenance)):
            x['output'][i]['provenance'] = original_provenance[i]
        return x
    
    dataset = dataset.map(add_missing_data, fn_kwargs=dict(original_dataset=original))

    return dataset

def trivia_filter(dataset, dataset_split):
    trivia_qa = datasets.load_dataset('trivia_qa', 'unfiltered.nocontext')[dataset_split]
    trivia_qa_ids = set(trivia_qa['question_id'])
    dataset = dataset.filter(lambda l: l['id'] in trivia_qa_ids)
    return dataset

for name in dataset_names:
    dataset = datasets.load_dataset('kilt_tasks', name )[split]
    if name == 'eli5':
        dataset = add_data_eli5(dataset)
    if name == 'triviaqa_support_only':
        print('called')
        dataset = trivia_filter(dataset, split)
    out_file = get_oracle_ranking_filename(run_folder, f'kilt_{name}'.replace('triviaqa_support_only', 'triviaqa'), split_mapping[split])
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
                
