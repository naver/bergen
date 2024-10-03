import argparse
import json
from pathlib import Path
from omegaconf import OmegaConf
import pandas as pd
import os 




def get_info(file):
    res= json.load( open(file))
    nbres=len(res)
    llen=[]
    nbsub=0
    for ex in res:
        llen.append(len(ex['Pred'].split(' ')))
        if ex['SUBSTR']:nbsub+=1 
    
    return nbres,nbsub/nbres,sum(llen)/nbres


def get_em_score(file):
    res= json.load( open(file))
    return res['em']


def get_bem_score(file):
    with open(file) as fd:
        return float(fd.readline().strip())

def get_config(file, split):
    config = OmegaConf.load(file)
    dataset_doc = config['dataset'][split]['doc']['init_args']['_target_'].replace('modules.dataset_processor.', '')
    dataset_query = config['dataset'][split]['query']['init_args']['_target_'].replace('modules.dataset_processor.', '')
    retriever = config['retriever']['init_args']['model_name'] if 'retriever' in config and 'init_args' in config['retriever'] else None
    reranker = config['reranker']['init_args']['model_name'] if 'reranker' in config and 'init_args' in config['reranker'] else None
    generator = config['generator']['init_args']['model_name'] if 'generator' in config and 'init_args' in config['generator'] else None
    prompt = config['prompt'] if 'prompt' in config else ''

    retrieve_top_k = config['retrieve_top_k'] if 'retriever' in config else '-'
    rerank_top_k = config['rerank_top_k'] if 'reranker' in config else '-'
    return dataset_query, dataset_doc, retriever, reranker, generator, prompt, retrieve_top_k, rerank_top_k

def get_scores(file, decimals=2):
    data = json.load(open(file))
    return data

def get_generation_time(file):
    data = json.load(open(file))
    return data['Generation time']

def get_ranking_metrics(file):
    data = json.load(open(file))
    return data['P_1']


def main(args):
    
    folder_path = Path(args.folder)
    ltuple=[]
    split = args.split
    for current_folder in folder_path.iterdir():
        skip = False
        try:
            if current_folder.is_dir() and not 'tmp_' in str(current_folder):
                gen_time = None
                ranking_metric = None
                files = [f.name for f in current_folder.iterdir()]

                if f'eval_{split}_metrics.json' in files:
                    x={}
                    for file_in_subfolder in current_folder.iterdir():
                        # try:
                        
                        if 'config.yaml' in str(file_in_subfolder):
                            dataset_query, dataset_doc, retriever, reranker, generator, prompt, retrieve_top_k, rerank_top_k = get_config(file_in_subfolder, split)

                            #preprocess the generator name,retriever,reranker name
                            generator_basename = os.path.basename(generator)
                            retriever_basename = os.path.basename(retriever)
                            reranker_basename = os.path.basename(reranker)
                            
                            x['exp_folder']=current_folder.name
                            x['query_dataset']=dataset_query.split('.')[-1]
                            x['Retriever']=retriever_basename
                            x['Reranker']=reranker_basename
                            x['Generator']=generator_basename
                                                    
                        if f'eval_{split}_metrics.json' in str(file_in_subfolder):
                            #m, em, f1, precision, recall, rouge1, rouge2, rougel, bem, LLMeval= get_scores(file_in_subfolder)
                            x_s=get_scores(file_in_subfolder)
                            x.update(x_s)
                        if f'eval_{split}_generation_time.json' in str(file_in_subfolder) :
                            gen_time = get_generation_time(file_in_subfolder) 
                            x['gen_time']=gen_time

                        if f'eval_{split}_ranking_metrics.json' in str(file_in_subfolder) :
                            ranking_metric = get_ranking_metrics(file_in_subfolder)
                            x['P_1']=ranking_metric
                        # except:
                        #     print(f'Failed to load {current_folder}!')  
                    
                    ltuple.append(x)
                    
        except Exception as e:
            print(f'Skipping {current_folder} due to parsing errors!')
            print(e)
    
    if len(ltuple) == 0:
        print(f'No results in folder "{args.folder}" yet!')
        exit()
    df= pd.DataFrame(ltuple)
    col=list(df.columns)
    llmeval_col = [c for c in col if 'llmeval' in c.lower()]

    if args.format =='simple':
        sel_col = ['exp_folder', 'query_dataset', 'Generator', 'Retriever', 'Reranker', "M", "EM", "Recall"] + llmeval_col
    elif args.format =='tiny':
        sel_col =['exp_folder', 'query_dataset', 'Generator', 'Retriever', 'Reranker', "M"] + llmeval_col
    elif args.format =='full':
        sel_col = ['exp_folder', 'Retriever', 'P_1', 'Reranker', 'Generator',  'gen_time', 'query_dataset', "M", "EM", "F1", "Precision", "Recall","Recall_char3gram", "Rouge-L"]+llmeval_col
    else:
        raise ValueError('Invalid output format')
    df=df[sel_col]
    
   
    df=df.sort_values(by=[args.sort])
    print('Split:', args.split)
    print(df.to_markdown(floatfmt=".2f"))
    if args.csv:
        os.makedirs('results', exist_ok=True)
        file_name = args.folder.replace('/', '_')
        df.to_csv(f'results/{file_name}.csv', index=False)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--folder", type=str, default='experiments')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument("--format", type=str, default='simple',choices=['simple', 'tiny', 'full'],
                        help='tiny prints Match and LLMEval; simple adds EM, R, Rg-L, BEM ; full prints all metrics and data')
    parser.add_argument("--sort", type=str, default="Generator")
    parser.add_argument("--csv", action='store_true')

    args = parser.parse_args()
    main(args)
