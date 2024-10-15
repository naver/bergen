import json
import shutil
import torch
import time
import os 
from hydra.utils import instantiate
import omegaconf
import yaml
import gc
import pandas as pd
pd.set_option("display.precision", 4)

class Evaluate:
    @staticmethod
    def eval(experiment_folder="experiments/", split="dev", bem: bool=False, llm: list[str]=None, llm_ollama: list[str]=None, vllm: list[str]=None, gpt: bool=None, bem_batch_size: int=1, lid: bool=None, lid_advanced: bool=None, llm_batch_size: int=None, llm_prompt: str = "default_qa", ollama_url: str=None, folder: str=None, force: bool=False, samples: int=-1):
        def eval_single(experiment_folder, folder, split: str, model, metric_name: str, nb_samples: int =-1):
            if folder != None:
                folders = [folder]
            else:
                folders = [ f.path for f in os.scandir(experiment_folder) if f.is_dir() and 'tmp_' not in f.path]
            for experiment_folder in folders:
    
                print('evaluating', experiment_folder)
                def load_data(input_file):
                    result_dict = json.load(open(input_file))
                    return pd.DataFrame(result_dict)
                
                input_file = f'{experiment_folder}/eval_{split}_out.json'
                if os.path.exists(input_file):
                    data = load_data(input_file)
                    if nb_samples >0 and nb_samples < len(data):
                        data = data[:nb_samples]
                                            
                    metrics_file = f'{experiment_folder}/eval_{split}_metrics.json'
                    try:
                       metrics_dict = json.load(open(metrics_file))
                    except: continue

                    if metric_name in metrics_dict and not force:
                        print (f"{experiment_folder}\t{metric_name}\talready done")
                        continue
                    
                    predictions = data['response'].values
                    references = data['label'].values
                    questions = data['question'].values    
                
                    if gpt is not None:
                        # openai costs
                        model_score, scores, cost = model(predictions, references, questions)
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
                    if nb_samples >0:
                        metric_name = f"{metric_name}_{nb_samples}"           
                    metrics_dict.update({metric_name: model_score})
                    print(metric_name,model_score)
                    # save to _ tmp file
                    with open(metrics_file + '_', 'w') as fp:
                        json.dump(metrics_dict, fp, indent=2)
                    # when writing successful remove tmp file
                    shutil.move(metrics_file + '_', metrics_file)
    
        if bem:
            from models.evaluators.bem import BEM
            model = BEM(batch_size=bem_batch_size)
            eval_single(experiment_folder, folder, split, model, 'BEM', nb_samples = samples)
        if gpt is not None:
            from models.evaluators.openai import OpenAI
            model = OpenAI(gpt)
            eval_single(experiment_folder, folder, split, model, gpt, nb_samples = samples)
        
        if llm is not None:
            
            if len(llm) == 0:
                model_config, short_name = "SOLAR-107B", "LLMeval"            
            elif len(llm)==1:
                model_config = llm[0]
                short_name = model_config
                short_name = f"LLMeval_{short_name}"        
            elif len(llm)==2:
                model_config = llm[0]
                short_name = llm[1]
                short_name = f"LLMeval_{short_name}"

            model_config = omegaconf.OmegaConf.load(f"config/generator/{model_config}.yaml")            
            if model_config['init_args']['_target_']=='models.generators.vllm.LLM':
                from models.evaluators.vllm import VLLMeval 
                model = VLLMeval(model_config, batch_size=llm_batch_size, config=llm_prompt)
                
            else:
                from models.evaluators.llm import LLMeval 
                model = LLMeval(model_config, batch_size=llm_batch_size, config=llm_prompt)
                if model.use_logits :
                    short_name = f"{short_name}_logits"
                
            eval_single(experiment_folder, folder, split, model, short_name, nb_samples = samples)
            del model
            torch.cuda.empty_cache()
            gc.collect()
        if llm_ollama is not None:
            from models.evaluators.llm_ollama import OllamaEval
            
            if len(llm_ollama)==1:
                model_config = llm_ollama[0]
                short_name = model_config
                short_name = f"LLMeval_{short_name}"        
            elif len(llm_ollama)==2:
                model_config = llm_ollama[0]
                short_name = llm_ollama[1] 
                short_name = f"LLMeval_{short_name}"
            if llm_batch_size == None:
                llm_batch_size = 1        
            model = OllamaEval(model_config, batch_size=llm_batch_size, config=llm_prompt, basic_url=ollama_url)
            eval_single(experiment_folder, folder, split, model, short_name, nb_samples = samples)
            
        if lid is not None or lid_advanced is not None:
            from models.evaluators.lid import LID
            from models.evaluators.lid_advanced import LID_advanced
            if folder == None:
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
                    model=LID(tgt_lng)  
                    eval_single(experiment_folder, folder, split, model, "lid", nb_samples = samples)
                if lid_advanced is not None:
                    model = LID_advanced(tgt_lng)
                    eval_single(experiment_folder, folder, split, model, "lid_advanced", nb_samples = samples)
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_folder', type=str, default="experiments/")
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--sample', type=int, default=-1, help="Use only subsample of the experiment folder for evaluation, useful for debug purposes (default -1: use full dataset)")    
    parser.add_argument('--bem', action='store_true')
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
    parser.add_argument('--bem_batch_size', type=int, default=1024)
    parser.add_argument('--llm_batch_size', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--llm_prompt', type=str, default="default_qa", help="Provide yaml config file with updated prompt. Default prompt: config/evaluator/default_prompt.yaml")
    parser.add_argument('--ollama_url', type=str, default="http://localhost:11434", help="")

    
    args = parser.parse_args()
    e = Evaluate.eval(
        folder=args.folder, 
        experiment_folder=args.experiments_folder, 
        split=args.split, 
        bem=args.bem,
        llm=args.llm, 
        llm_ollama=args.llm_ollama,
        gpt=args.gpt,
        lid=args.lid,
        lid_advanced=args.lid_advanced,
        bem_batch_size=args.bem_batch_size,
        llm_batch_size=args.llm_batch_size,
        llm_prompt=args.llm_prompt,
        ollama_url=args.ollama_url,
        force=args.force, 
        samples=args.sample
    )

