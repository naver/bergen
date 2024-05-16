
#experiments/5bf32d6105d324d6/eval_dev_ranking_metrics.json

import sys
sys.path.append('../')
from utils import eval_retrieval_kilt, get_ranking_filename,load_trec
import os
from omegaconf import OmegaConf
import json

experiments_folder = '../experiments/'


qrels_folder = '../qrels/'
runs_folder = '../runs/'

get_clean_model_name = lambda l: l.replace('/', '_')

doc_dataset_name = 'kilt-100w'
dataset_split = 'dev'
for experiment_folder in os.listdir(f'{experiments_folder}'):
    if not experiment_folder.startswith('tmp_'):
        print(experiment_folder)
        config = OmegaConf.load(f'{experiments_folder}/{experiment_folder}/config.yaml')
        dataset = config['dataset'][dataset_split]['query']['init_args']['_target_']

        generation_top_k = config['generation_top_k']
        retrieve_top_k = config['retrieve_top_k']
        if 'retriever' in config:
            experiment_folder = f'{experiments_folder}/{experiment_folder}'
            retriever_name = get_clean_model_name(config['retriever']['init_args']['model_name'])
            if 'oracle_provenance' == retriever_name:
                metrics = {"P_1": 1.0, f"recall_{generation_top_k}": 1.0}
                open(f'{experiment_folder}/eval_{dataset_split}_ranking_metrics.json', 'w').write(json.dumps(metrics))
            else:
                query_ids, doc_ids, scores = load_trec(f'{experiment_folder}/eval_{dataset_split}_ranking_run.trec')
                if 'Eli5' in dataset:
                    query_dataset_name = 'kilt_eli5'
                elif 'Wow' in dataset:
                    query_dataset_name = 'kilt_wow'
                elif 'Hotpot' in dataset:
                    query_dataset_name = 'kilt_hotpotqa'
                elif 'Trivia' in dataset:
                    query_dataset_name = 'kilt_triviaqa'
                elif 'NQ' in dataset:
                    query_dataset_name = 'kilt_nq'

                eval_retrieval_kilt(
                    experiment_folder, 
                    qrels_folder, 
                    query_dataset_name, 
                    dataset_split, 
                    query_ids, 
                    doc_ids, 
                    scores, 
                    top_k=generation_top_k,
                    write_trec=False
                    )
        

