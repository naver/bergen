import json
import shutil
import torch
import os 
import omegaconf
import yaml
import gc
import pandas as pd
pd.set_option("display.precision", 4)


def load_data(input_file, nb_samples):
    result_dict = json.load(open(input_file))
    data = pd.DataFrame(result_dict)
    if nb_samples > 0 and nb_samples < len(data):
        data = data[:nb_samples]
    return data


def eval_single(experiment_folder,
                folder,
                split: str,
                model,
                metric_name: str,
                nb_samples: int = -1,
                gpt: str = None,
                win_rate_opponent_folder: str = None,
                force: bool = False,
                ):
    if nb_samples >0:
        metric_name = f"{metric_name}_{nb_samples}"
    if folder is not None:
        folders = [folder]
    else:
        folders = [ f.path for f in os.scandir(experiment_folder) if f.is_dir() and 'tmp_' not in f.path]
    for experiment_folder in folders:
        print('evaluating', experiment_folder)
        
        input_file = f'{experiment_folder}/eval_{split}_out.json'
        if os.path.exists(input_file):
            data = load_data(input_file, nb_samples=nb_samples)
                                    
            metrics_file = f'{experiment_folder}/eval_{split}_metrics.json'
            try:
                metrics_dict = json.load(open(metrics_file))
            except: 
                continue

            if (metric_name in metrics_dict or metric_name + '_tie' in metrics_dict) and not force:
                print (f"{experiment_folder}\t{metric_name}\talready done")
                continue
            
            predictions = data['response'].values
            references = data['label'].values
            questions = data['question'].values    
        
            if gpt is not None:
                if win_rate_opponent_folder is None:
                    model_score, scores, cost = model(predictions, references, questions)
                else:
                    # We filter the other data to keep the q_ids in data
                    other_data = load_data(f'{win_rate_opponent_folder}/eval_{split}_out.json', nb_samples=-1)
                    other_data = other_data[other_data.q_id.isin(data.q_id.unique())]
                    # Reordering along data order:
                    other_data = other_data.set_index('q_id').reindex(data['q_id']).reset_index()

                    # Sanity checks:
                    for elt, other_elt in zip(data['q_id'].values, other_data['q_id'].values):
                        assert elt == other_elt, f'Unmatching q_id {elt} vs {other_elt} in json files: cannot compare'
                    other_predictions = other_data['response'].values

                    model_score, scores, cost = model.pairwise_win_rate(predictions, other_predictions, references, questions)
                    
                # openai costs
                costs_out_file = f'{experiment_folder}/eval_{split}_cost_{metric_name}_out.json'
                with open(costs_out_file, 'w') as fout: fout.write(json.dumps(cost))
            else:                    
                model_score, scores = model(predictions, references, questions)
            data[metric_name] = scores
            metrics_out_file = f'{experiment_folder}/eval_{split}_out.json'
            if nb_samples >0:
                metrics_out_file = f'{experiment_folder}/eval_{split}_out_{nb_samples}.json'
                
            # temporary print eval_out results with updated metric  (to avoid loosing eval_dev_out.json if smth goes wrong)                   
            data.to_json(metrics_out_file+"_", orient='records') 
            #move temprorary result into final name                       
            shutil.move(metrics_out_file + '_', metrics_out_file)
            if isinstance(model_score, dict): # win tie lose for pairwise !
                metrics_dict.update({metric_name + '_' + k: v for k, v in model_score.items()})
            else:
                metrics_dict.update({metric_name: model_score})
            print(metric_name, model_score)
            # save to _ tmp file
            with open(metrics_file + '_', 'w') as fp:
                json.dump(metrics_dict, fp, indent=2)
            # when writing successful remove tmp file
            shutil.move(metrics_file + '_', metrics_file)
                    
                    
def llm_eval(llm: list[str], experiment_folder, folder, split, batch_size, llm_prompt, nb_samples, force):
    if len(llm) == 0:
        model_config, short_name = "SOLAR-107B", "LLMeval"            
    else:
        model_config = llm[0]
        short_name = llm[1] if len(llm) > 1 else model_config
        short_name = f"LLMeval_{short_name}"

    model_config = omegaconf.OmegaConf.load(f"config/generator/{model_config}.yaml")            
    if model_config['init_args']['_target_']=='models.generators.vllm.VLLM':
        from models.evaluators.vllm import VLLMeval 
        model = VLLMeval(model_config, batch_size=batch_size, config=llm_prompt)
        
    else:
        from models.evaluators.llm import LLMeval 
        model = LLMeval(model_config, batch_size=batch_size, config=llm_prompt)
        if model.use_logits :
            short_name = f"{short_name}_logits"
        
    eval_single(experiment_folder, folder, split, model, metric_name=short_name, nb_samples=nb_samples, force=force)
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    
def llm_ollama_eval(llm_ollama: list[str], experiment_folder, folder, split, batch_size, llm_prompt, ollama_url, nb_samples, force):
    from models.evaluators.llm_ollama import OllamaEval
    
    if len(llm_ollama) > 0:
        model_config = llm_ollama[0]
        short_name = llm_ollama[1] if len(llm_ollama) > 1 else model_config
        short_name = f"LLMeval_{short_name}"
        
    batch_size = batch_size or 1  
            
    model = OllamaEval(model_config, batch_size=batch_size, config=llm_prompt, basic_url=ollama_url)
    eval_single(experiment_folder, folder, split, model, metric_name=short_name, nb_samples = nb_samples, force=force)
    

def lid_eval(lid, lid_advanced, experiment_folder, folder, split, nb_samples, force):
    from models.evaluators.lid import LID
    from models.evaluators.lid_advanced import LID_advanced
    if folder is None:
        folders = [ f.path for f in os.scandir(experiment_folder) if f.is_dir() and 'tmp_' not in f.path]
    else:
        folders = [folder]

    for folder in folders:
        # we need to get language from each folder config separately
        config = yaml.safe_load(open(f"{folder}/config.yaml")) 
        if 'lng' in config['dataset'][split]['query']['init_args']:
            tgt_lng = config['dataset'][split]['query']['init_args']['lng']
        elif  'lang' in  config['dataset'][split]['query']['init_args']:
            tgt_lng = config['dataset'][split]['query']['init_args']['lang']
        else:
            #if language is not specified we set it to English by default
            tgt_lng = 'en'
            print(f"{folder}: didn't find lng in the config.yaml, set it to English by default")
        if lid is not None:
            model = LID(tgt_lng)  
            eval_single(experiment_folder, folder, split, model, metric_name="lid", nb_samples = nb_samples, force=force)
        if lid_advanced is not None:
            model = LID_advanced(tgt_lng)
            eval_single(experiment_folder, folder, split, model, metric_name="lid_advanced", nb_samples = nb_samples, force=force)
            
            
def gpt_eval(gpt, experiment_folder, folder, split, win_rate_opponent_folder, win_rate_opponent_name, nb_samples, force):
    from models.evaluators.openai import OpenAI
    model = OpenAI(gpt)
    metric_name = gpt
    if win_rate_opponent_folder is not None:
        metric_name += '_win_rate_' + win_rate_opponent_name
    eval_single(experiment_folder, folder, split, model, gpt=gpt, metric_name=metric_name, nb_samples=nb_samples, win_rate_opponent_folder=win_rate_opponent_folder, force=force)


def run_eval(experiment_folder="experiments/",
             split="dev",
             llm: list[str]=None,
             llm_ollama: list[str]=None,
             vllm: list[str]=None,
             gpt: bool=None,
             lid: bool=None,
             lid_advanced: bool=None,
             llm_batch_size: int=None,
             llm_prompt: str = "default_qa",
             ollama_url: str=None,
             folder: str=None,
             force: bool=False,
             nb_samples: int=-1,
             win_rate_opponent_folder: str = None,
             win_rate_opponent_name: str = None):
        if gpt is not None:
            gpt_eval(gpt, 
                     experiment_folder, 
                     folder, 
                     split, 
                     win_rate_opponent_folder=win_rate_opponent_folder, 
                     win_rate_opponent_name=win_rate_opponent_name, 
                     nb_samples=nb_samples, 
                     force=force)
        
        if llm is not None:
            llm_eval(llm, experiment_folder, folder, split, llm_batch_size, llm_prompt, nb_samples=nb_samples, force=force)
            
        if llm_ollama is not None:
            llm_ollama_eval(llm_ollama, experiment_folder, folder, split, llm_batch_size, llm_prompt, ollama_url, nb_samples=nb_samples, force=force)
            
        if lid is not None or lid_advanced is not None:
            lid_eval(lid, lid_advanced, experiment_folder, folder, split, nb_samples=nb_samples, force=force)
            

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_folder', type=str, default="experiments/")
    parser.add_argument('--folder', type=str, default=None)
    
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--sample', type=int, default=-1, help="Use only subsample of the experiment folder for evaluation, useful for debug\
        purposes (default -1: use full dataset)")    
    parser.add_argument('--lid', action='store_true', default=None)
    parser.add_argument('--lid_advanced', action='store_true', default=None)

    parser.add_argument('--llm', type=str, nargs='*', default=None, 
            help=""" 
                - full model name (corresponding to generator config name) and short name (used for naming output files and metrics): 
                    eg. -llm SOLAR-107B solar 
                - if short name is missing: use full name in naming, 
                - if no arguments specified: falls back to default arguments: uses default values (SOLAR-107B LLMeval). 
                """)
                    
    parser.add_argument('--llm_ollama',  type=str, nargs='*', default=None, 
            help="""
                Calls ollama server to run evaluation. Requires 1 or 2 arguments: 
                - full model name  and short name (used for naming output files and metrics): eg. -llm_ollama llama3:default llama3 
                - if short name is missing: use full name in naming
                """ )
    parser.add_argument('--gpt', type=str,default=None)
    parser.add_argument('--win_rate_opponent_folder', type=str, default=None, help='Provide a second folder via this to run pairwise comparisons\
        (only available with gpt and when specifying a folder)')
    parser.add_argument('--win_rate_opponent_name', type=str, default=None, help='Provide a second folder via this to run pairwise comparisons\
        (only available with gpt and when specifying a folder)')
    
    parser.add_argument('--llm_batch_size', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--llm_prompt', type=str, default="default_qa", help="Provide yaml config file with updated prompt.\
        Default prompt: config/evaluator/default_prompt.yaml")
    parser.add_argument('--ollama_url', type=str, default="http://localhost:11434", help="")
    
    args = parser.parse_args()
    
    if args.win_rate_opponent_folder is not None:
        assert args.gpt is not None, 'Pairwise only supported with gpt currently'
        assert args.folder is not None, 'Pairwise only supported if you specify a folder'
        assert os.path.isdir(args.win_rate_opponent_folder), 'Pairwise_on argument should point to a directory to which compare the folder arg outputs.'
        assert args.win_rate_opponent_name is not None, 'Specify a name for the opponent'
        print('Pairwise comparison detected:', args.win_rate_opponent_folder, args.win_rate_opponent_name)
    
    e = run_eval(
        folder=args.folder, 
        experiment_folder=args.experiments_folder, 
        split=args.split, 
        llm=args.llm, 
        llm_ollama=args.llm_ollama,
        gpt=args.gpt,
        lid=args.lid,
        lid_advanced=args.lid_advanced,
        llm_batch_size=args.llm_batch_size,
        llm_prompt=args.llm_prompt,
        ollama_url=args.ollama_url,
        force=args.force, 
        nb_samples=args.sample,
        win_rate_opponent_folder=args.win_rate_opponent_folder,
        win_rate_opponent_name=args.win_rate_opponent_name
    )

