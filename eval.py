import json
import shutil
import torch
import time
import os 

class Evaluate:
    @staticmethod
    def eval(experiment_folder, split, bem=False, llm=False, vllm=False,gpt=None,clova=False,bem_batch_size=1, llm_batch_size=1, folder=None, force=False):
        def eval_single(experiment_folder, folder, split, model, metric_name):
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
                    
                    predictions, references, questions = list(), list(), list()

                    for sample in data:
                        predictions.append(sample['response'])
                        references.append(sample['label'])
                        questions.append(sample['question'])

                    if gpt is not None:
                        # openai costs
                        model_score, scores, cost = model(predictions, references, questions)
                        costs_out_file = f'{experiment_folder}/eval_{split}_cost_{metric_name}_out.json'
                        with open(costs_out_file, 'w') as fout: fout.write(json.dumps(cost))
                    else:
                        model_score, scores = model(predictions, references, questions)

                    metrics_out_file = f'{experiment_folder}/eval_{split}_metrics_{metric_name}_out.json'
                    with open(metrics_out_file, 'w') as fout:

                        for score, sample in zip(scores, data):
                            jsonl = {'question' : sample['question'], 'response': sample['response'], 'label': sample['label'], 'score': float(score)}
                            fout.write(json.dumps(jsonl)+'\n')
                            
                    metrics_dict.update({metric_name: str(model_score)})
                    print (metrics_dict,metric_name,model_score)
                    # save to _ tmp file
                    with open(metrics_file + '_', 'w') as fp:
                        json.dump(metrics_dict, fp)
                    # when writing successful remove tmp file
                    shutil.move(metrics_file + '_', metrics_file)
    

        if bem:
            from models.evaluators.bem import BEM
            model = BEM(batch_size=bem_batch_size)
            eval_single(experiment_folder, folder, split, model, 'BEM')
        if llm:
            from models.evaluators.llm import LLM
            model_name, short_name = "Upstage/SOLAR-10.7B-Instruct-v1.0", "LLMeval"
            model = LLM(model_name, batch_size=llm_batch_size)
            
            eval_single(experiment_folder, folder, split, model, short_name)
        
        if gpt is not None:
            from models.evaluators.openai import OpenAI
            model = OpenAI(gpt)
            eval_single(experiment_folder, folder, split, model, gpt)

        if vllm:
            from models.evaluators.vllm import LLM
            model_name, short_name = "Upstage/SOLAR-10.7B-Instruct-v1.0", "LLMeval"
            model = LLM(model_name, batch_size=llm_batch_size)
            eval_single(experiment_folder, folder, split, model, short_name)

    

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--experiments_folder', type=str, default="experiments/")
    parser.add_argument('--folder', type=str, default=None)
    parser.add_argument('--split', type=str, default='dev')
    parser.add_argument('--bem', action='store_true')
    parser.add_argument('--llm', action='store_true')
    parser.add_argument('--vllm', action='store_true')
    parser.add_argument('--gpt', type=str,default=None)
    parser.add_argument('--clova', action='store_true')
    parser.add_argument('--bem_batch_size', type=int, default=1024)
    parser.add_argument('--llm_batch_size', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    e = Evaluate.eval(
        args.experiments_folder, 
        args.split, 
        bem=args.bem,
        llm=args.llm, 
        vllm=args.vllm, 
        gpt=args.gpt,
        clova=args.clova,
        bem_batch_size=args.bem_batch_size,
        llm_batch_size=args.llm_batch_size,
        folder=args.folder, 
        force=args.force
        )
