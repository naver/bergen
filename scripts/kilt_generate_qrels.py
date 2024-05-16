import datasets
import os
import sys
sys.path.append('../')
from utils import get_qrel_ranking_filename
import json
from collections import defaultdict
import urllib

dataset_names = ['aidayago2', 'cweb', 'eli5', 'fever', 'hotpotqa', 'nq', 'structured_zeroshot', 'trex', 'triviaqa_support_only', 'wned', 'wow']
qrel_folder = '../qrels'
os.makedirs(qrel_folder, exist_ok=True)
splits = ['validation']
def add_missing_data_eli5(dataset):
    # hf dataset is missing provenances add them
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
        
    return dataset.map(add_missing_data, fn_kwargs=dict(original_dataset=original))


split_mapping = {'validation': 'dev', 'test': 'test', 'training': 'train'}
qrels = defaultdict(dict)
for split in splits:
    for name in dataset_names:
        dataset = datasets.load_dataset('kilt_tasks', name )[split]
        out_file = get_qrel_ranking_filename(qrel_folder, f'kilt_{name}'.replace('triviaqa_support_only', 'triviaqa'), split_mapping[split])
        if 'eli5' in name:
            print('Adding missing data to ElI5')
            dataset = add_missing_data_eli5(dataset)
        qrel_trec_out = open(out_file.replace('.json', '.txt'), 'w')
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
                
