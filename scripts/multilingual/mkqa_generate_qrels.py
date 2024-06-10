import os
dir_path = os.path.dirname(os.path.realpath(__file__))

import sys
sys.path.append(dir_path+'/../..')

import datasets
from utils import get_qrel_ranking_filename
import json
from collections import defaultdict
import urllib

lng = sys.argv[1]
lng_mkqa = lng if lng != "zh" else "zh_cn"
name = f"mkqa_{lng}" # dataset name from module/dataset_processor.py

# output filenames
qrel_folder = dir_path + '/../../qrels'
os.makedirs(qrel_folder, exist_ok=True)
out_file = get_qrel_ranking_filename(qrel_folder, name, "dev")
qrel_trec_out = open(out_file.replace('.json', '.txt'), 'w')

qrels = defaultdict(dict)
qrels["doc_dataset_name"] = "kilt-100w" # oracle provenance in mkqa is provided for kilt-100w (in English)
# in case mkqa is run with other wiki, e.g. in non-English, retrieval evaluation will be skipped

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

# write qrels
for sample in dataset:
    query_id = sample['id']
    paragraph_bounds, wiki_ids = list(), list()
    for out in sample['output']:
        if len(out['answer']) > 0 and 'provenance' in out:
            wiki_id_out = [prov['wikipedia_id'] for prov in out['provenance']]
            #print(wiki_id_out)
            wiki_ids += wiki_id_out
    wiki_ids = list(set(wiki_ids))
    for wiki_id in wiki_ids:
        qrel_trec_out.write(f'{query_id} 0 {wiki_id} 1\n')
        qrels[query_id].update({wiki_id  : 1})
qrel_trec_out.close()

with open(out_file, 'w') as fp:
    json.dump(qrels, fp)
                
