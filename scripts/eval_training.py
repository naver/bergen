import os 
from omegaconf import OmegaConf
import json
import pandas as pd
experiments_folder = 'experiments_emb'
tuples = list()
for experiment_folder in os.listdir(f'{experiments_folder}'):
    
    if not experiment_folder.startswith('tmp_'):
        config = OmegaConf.load(f'{experiments_folder}/{experiment_folder}/config.yaml')
        retr = config.retriever.init_args.model_name if 'retriever' in config else None
        rer = config.reranker.init_args.model_name if 'reranker' in config else None
        gen = config.generator.init_args.model_name
        lr = config.train.trainer.learning_rate if 'train' in config else None
        m = json.loads(open(f'{experiments_folder}/{experiment_folder}/eval_dev_metrics.json').read())['EM']
        tuples.append([experiment_folder, retr, rer, gen, lr, m ])

df = pd.DataFrame(tuples)
df.columns = ['Folder', 'Retriever', 'Reranker', 'Generator', 'Learning Rate', 'EM']

print(df.to_markdown())
