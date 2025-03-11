import json
import shutil
import torch
import time
import os 
from hydra.utils import instantiate
import omegaconf
import yaml
from torch.profiler import profile, record_function, ProfilerActivity
import gc

class Evaluate:
    @staticmethod
    def eval(experiment_folder="experiments/", split="dev", bem=False, llm=None, llm_ollama=None, vllm=None,gpt=None,bem_batch_size=1, lid=None, lid_advanced=None, llm_att: bool=False, llm_batch_size=1, llm_prompt="default_qa", ollama_url=None, folder=None, force=False, samples=None):
        def eval_single(experiment_folder, folder, split, model, metric_name, nb_samples: int = -1):
            if folder != None:
                folders = [folder]
            else:
                folders = [ f.path for f in os.scandir(experiment_folder) if f.is_dir() and 'tmp_' not in f.path]
            for experiment_folder in folders:
    
                print('evaluating', experiment_folder)
                def load_data(input_file):
                    result_dict = json.load(open(input_file))
                    return result_dict
                
                input_file = f'{experiment_folder}/eval_{split}_out.json'
                if os.path.exists(input_file):
                    data = load_data(input_file)
                    metrics_file = f'{experiment_folder}/eval_{split}_metrics.json' 
                    try:
                       metrics_dict = json.load(open(metrics_file))
                    except: continue

                    if metric_name in metrics_dict and not force:
                        print (f"{experiment_folder}\t{metric_name}\talready done")
                        continue
                    
                    predictions, references, questions, instructions = list(), list(), list(), list()

                    for sample in data:
                        predictions.append(sample['response'])
                        references.append(sample['label'])
                        questions.append(sample['question'])
                        instructions.append(sample['instruction'])
                        if nb_samples is not None and len(predictions)==nb_samples:
                            break

                    if gpt is not None:
                        # openai costs
                        model_score, scores, cost = model(predictions, references, questions)
                        costs_out_file = f'{experiment_folder}/eval_{split}_cost_{metric_name}_out.json'
                        with open(costs_out_file, 'w') as fout: fout.write(json.dumps(cost))
                    else:                    
                        if metric_name == "att":
                            model_score, scores = model(predictions, references, questions, instructions)
                        else:
                            model_score, scores = model(predictions, references, questions)
                        # model_score, scores = model(predictions, references, questions)
                    if metric_name =="att":
                        # metrics_score is a dict of different att metrics in this case
                        if nb_samples > 0:
                            metrics_dict.update(model_score)      
                        else:
                            metrics_dict.update({f'{k}_{nb_samples}':model_score[k] for k in model_score})      
                        # print(len(scores))
                        # print(len(data))
                        # for k in scores:
                        #     data[k] = scores[k]
                        # pass
                    else:
                        metrics_out_file = f'{experiment_folder}/eval_{split}_metrics_{metric_name}_out.json'
                        with open(metrics_out_file, 'w') as fout:

                            for score, sample in zip(scores, data):
                                jsonl = {'question' : sample['question'], 'response': sample['response'], 'label': sample['label'], 'score': float(score)}
                                fout.write(json.dumps(jsonl)+'\n')
                                
                        metrics_dict.update({metric_name: str(model_score)})
                        
                    print(metric_name,model_score)
                    # save to _ tmp file
                    with open(metrics_file + '_', 'w') as fp:
                        json.dump(metrics_dict, fp, indent=2)
                    # when writing successful remove tmp file
                    shutil.move(metrics_file + '_', metrics_file)
    
        #with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
        #            profile_memory=True, record_shapes=True) as prof:
        if bem:
            from models.evaluators.bem import BEM
            model = BEM(batch_size=bem_batch_size)
            eval_single(experiment_folder, folder, split, model, 'BEM', nb_samples = samples)
        if gpt is not None:
            from models.evaluators.openai import OpenAI
            model = OpenAI(gpt)
            eval_single(experiment_folder, folder, split, model, gpt, nb_samples = samples)
        
        if vllm is not None:
            from models.evaluators.vllm import VLLMeval 
            if len(vllm) == 0:
                # corresponds to default LLMeval setting, results reported in the paper
                model_config, short_name = "SOLAR-107B", "VLLMeval"             
            elif len(vllm)==1:
                model_config = vllm[0]
                short_name = f"VLLMeval_{model_config}"
            elif len(vllm)==2:
                model_config = vllm[0]
                short_name = f"VLLMeval_{vllm[1]}"        
            model = VLLMeval(model_config, batch_size=llm_batch_size, config=llm_prompt)
            eval_single(experiment_folder, folder, split, model, short_name, nb_samples = samples)
            del model
            torch.cuda.empty_cache()
            gc.collect()
        if llm is not None:
            from models.evaluators.llm import LLMeval
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
            model = LLMeval(model_config, batch_size=llm_batch_size, config=llm_prompt)
            if model.use_logits:
                short_name = f"{short_name}_logits"
            eval_single(experiment_folder, folder, split, model, short_name, nb_samples = samples)
            del model
            torch.cuda.empty_cache()
            gc.collect()
            
        if llm_att is not None :
            from models.evaluators.llm_att import LLM_att
            if folder == None:
                folders = [ f.path for f in os.scandir(experiment_folder) if f.is_dir() and 'tmp_' not in f.path]
            else:
                folders = [folder]
            model_name = ''
            for folder in folders:
                #get model name from  config
                config = yaml.safe_load(open(f"{folder}/config.yaml"))
                generator = config['generator']
                prompt = config['prompt']
                if not config['generator']['init_args']['model_name'] == model_name :
                    # try:
                    generator['init_args']['_target_'] = generator['init_args']['_target_'].replace('vllm', 'llm')
                    
                    model = LLM_att(generator, prompt)   

                    model_name = config['generator']['init_args']['model_name']

                    # except:
                    #     print("Skip", folder, model_name )
                    #     continue
                else:
                    #if other folder used the same generator, do not load it again, but update prompt
                    model.llm.model.prompt = prompt
                short_name = "att"
                eval_single(experiment_folder, folder, split, model, short_name)
                
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
            model = OllamaEval(model_config, batch_size=llm_batch_size, config=llm_prompt, basic_url=ollama_url)
            eval_single(experiment_folder, folder, split, model, short_name, nb_samples = samples)
            
        if lid is not None or lid_advanced is not None:
            from models.evaluators.lid import LID
            from models.evaluators.lid_advanced import LID_advanced
            if folder == None:
                folders = [ f.path for f in os.scandir(experiment_folder) if f.is_dir() and 'tmp_' not in f.path]
            else:
                folders = [folder]
                tgt_lng_lid = lid
                tgt_lng_lid1 = lid_advanced
            for folder in folders:
                # we need to get language from each folder config separately
                print(folder)
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
                    eval_single(experiment_folder, folder, split, model, "lid_debug")
                if lid_advanced is not None:
                    model = LID_advanced(tgt_lng)
                    eval_single(experiment_folder, folder, split, model, "lid_advanced")
        #print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_folder', type=str, default="experiments/")
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--sample', type=int, default=None, help="Use only subsample of the experiment folder for evaluation, useful for debug purposes")    
    parser.add_argument('--bem', action='store_true')
    parser.add_argument('--lid', action='store_true', default=None)
    parser.add_argument('--llm_att', action='store_true', default=None, help="Compute attention-based metrics")
    parser.add_argument('--lid_advanced', action='store_true', default=None)

    parser.add_argument('--llm', type=str, nargs='*', default=None, 
            help="""
                Uses default HF inference mechanism for LLM evaluation.  Requires up to 2 arguments: 
                - full model name  and short name (used for naming output files and metrics): eg. -llm \"Upstage/SOLAR-10.7B-Instruct-v1.0\" solar 
                - if short name is missing: use full name in naming, 
                - if no arguments specified: falls back to default arguments: uses default values (\"Upstage/SOLAR-10.7B-Instruct-v1.0\" solar). 
                """)
    parser.add_argument('--vllm', type=str, nargs='*', default=None, 
                help="""
                    Calls vllm to run evalution. Requires 2 arguments: 
                    Uses default HF inference mechanism for LLM evaluation.  Requires up to 2 arguments: 
                    - full model name  and short name (used for naming output files and metrics): eg. -vllm \"Upstage/SOLAR-10.7B-Instruct-v1.0\" solar 
                    - if short name is missing: use full name in naming, 
                    -  if no arguments specified: falls back to default arguments: uses default values (\"Upstage/SOLAR-10.7B-Instruct-v1.0\" solar). 
                """)
                    
    parser.add_argument('--llm_ollama',  type=str, nargs='*', default=None, 
            help="""
                Calls ollama server to run evaluation. Requires 1 or 2 arguments: 
                - full model name  and short name (used for naming output files and metrics): eg. -llm_ollama llama3:default llama3 
                - if short name is missing: use full name in naming
                """ )
    parser.add_argument('--gpt', type=str,default=None)
    parser.add_argument('--bem_batch_size', type=int, default=1024)
    parser.add_argument('--llm_batch_size', type=int, default=1)
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
        llm_att = args.llm_att,        
        llm_ollama=args.llm_ollama,
        vllm=args.vllm, 
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

