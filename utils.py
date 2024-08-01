'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import datasets
import random 
import json
from collections import defaultdict
import shutil
import pytrec_eval
from datasets.fingerprint import Hasher
import os 
from omegaconf import OmegaConf
import torch
import time
import glob
from tqdm import tqdm
import warnings
import numpy as np

# needed because HF dataset does not allow indexing by id (only index)
# given a set of ids return field in dataset, if no field provided just return indexes
def get_by_id(dataset, ids, field=None):
    # if single id is passed cast it to list
    if not isinstance(ids, list):
        ids = [ids]
    idxs = [ dataset.id2index[id_] for id_ in ids if id_ in dataset.id2index]
    if field != None:
        return dataset[idxs][field] if field in dataset[idxs] else []
    else:
        return idxs
    
def load_embeddings(index_path):
    try:
        emb_files = glob.glob(f'{index_path}/*.pt')
        sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
        embeds = list()
        for i, emb_file in enumerate(tqdm(sorted_emb_files, total=len(sorted_emb_files), desc=f'Load embeddings...')):
            emb_chunk = torch.load(emb_file)
            embeds.append(emb_chunk)
        embeds = torch.concat(embeds)
    except:
        raise IOError(f'Embedding index corrupt. Please delete folder "{index_path}" and run again.')
    return embeds

def get_embeddings_by_id(ids, index_path):
    if not isinstance(ids[0], list):
        ids = [ids]
    current_index  = 0
    emb_files = glob.glob(f'{index_path}/*.pt')
    sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
    embeds_dict = {}
    
    ids_flat = sum(ids, [])
    for emb_file in tqdm(sorted_emb_files, total=len(sorted_emb_files), desc=f'Getting embeddings...'):
        emb_chunk = torch.load(emb_file)
        emb_chunk_size = emb_chunk.shape[0]
        idx_in_chunk = list(set(range(current_index, current_index + emb_chunk_size)).intersection(ids_flat))
        rel_idx_in_chunk = [idx-current_index for idx in idx_in_chunk]
        if len(idx_in_chunk) > 0:
            embds_current_chunk = emb_chunk[rel_idx_in_chunk]
            for idx, emb in zip(idx_in_chunk, embds_current_chunk):
                embeds_dict[idx] = emb
        current_index += emb_chunk_size

    embeds = list()
    for ids_q in ids:
        embeds_q = torch.stack([embeds_dict[id_] for id_ in ids_q])
        embeds.append(embeds_q)
    embeds = torch.stack(embeds)
    return embeds

def get_doc_embeds_from_dataset(d_ids, embeds_dataset):
    if len(d_ids) == 0:
        return None
    doc_embeds = []
    # get doc embeddings
    for d_id_list in d_ids:
        embeds_list  = embeds_dataset[d_id_list]['embedding']
        embeds = torch.tensor(embeds_list).squeeze(1)
        doc_embeds.append(embeds)
    doc_embeds = torch.stack(doc_embeds) 

    return doc_embeds.cpu()

# horrible function :/ needs to be refactored into mult_doc and single doc
# gets q_ids and d_ids and does a lookup by id to get the content
# then constructs hf_dataset out of it
def prepare_dataset_from_ids(dataset, q_ids, d_ids, multi_doc=False, query_embeds_path=None, doc_embeds_path=None, query_field="content"):

    # if query _ids and doc_ids are None only return queries and optional labels /ranking labels
    if q_ids == d_ids == None:
        dataset_dict = {
            'query': dataset['query'][query_field], 
            'q_id': dataset['query']['id'],
            }
        # if labels or ranking_labels are in query dataset add them to the dataset

        dataset_dict.update({'label': dataset['query']['label'] } if 'label' in dataset['query'].features else {})
        dataset_dict.update({'ranking_label': dataset['query']['ranking_label']} if 'ranking_label' in dataset['query'].features else {})
    else:
        dataset_dict = defaultdict(list)
        # get labels
        labels = get_by_id(dataset['query'], q_ids, 'label')
        # get ranking_labels
        ranking_labels = get_by_id(dataset['query'], q_ids, 'ranking_label') 
        # get queries
        queries = get_by_id(dataset['query'], q_ids, query_field)
        # put together dataset_dict for each query
        for i, q_id in tqdm(enumerate(q_ids), desc='Fetching data from dataset...', total=len(q_ids)):
            docs = get_by_id(dataset['doc'], d_ids[i], 'content') 
            doc_idxs = get_by_id(dataset['doc'], d_ids[i])
            # for multi_doc=True, all documents are saved to the 'doc' entry

            if multi_doc:
                dataset_dict['doc'].append(docs)
                dataset_dict['query'].append(queries[i])
                dataset_dict['q_id'].append(q_id)
                dataset_dict['d_id'].append(d_ids[i])
                dataset_dict['d_idx'].append(doc_idxs)
                # add labels if they exist in datset
                if len(labels) > 0:
                    dataset_dict['label'].append(labels[i])
                # add ranking_labels if they exist in datset
                if len(ranking_labels) > 0:
                    dataset_dict['ranking_label'].append(ranking_labels[i])
            else:
                # for multi_doc=False, we save every document to a new entry
                    for d_id, doc, d_idx in zip(d_ids[i], docs, doc_idxs):
                        dataset_dict['d_id'].append(d_id)
                        dataset_dict['d_idx'].append(d_idx)
                        dataset_dict['doc'].append(doc)
                        dataset_dict['query'].append(queries[i])
                        dataset_dict['q_id'].append(q_id)
                        # add labels if they exist in datset
                        if len(labels) > 0 :
                            dataset_dict['label'].append(labels[i])
                        # add ranking_labels if they exist in datset
                        if len(ranking_labels) > 0:
                            dataset_dict['ranking_label'].append(ranking_labels[i])

    return datasets.Dataset.from_dict(dataset_dict)

def print_generate_out(queries, instructions, responses, query_ids, labels, ranking_labels, n=5):
    rand = random.sample(range(len(query_ids)), n)
    for i in rand:
        print('_'*50)
        print('Query ID:', query_ids[i])
        print('Query:', queries[i])
        print('_'*50)
        if instructions[i] != None:
            print('Instruction to Generator:')
            print(instructions[i])
        print()
        print('LLM Answer:')
        print(responses[i])
        print('Label(s):')
        print(labels[i])
        if ranking_labels[i] != None:
            print('Ranking Label(s):')
            print(ranking_labels[i])
        print()
        print()


def print_rag_model(rag, retriever_kwargs,reranker_kwargs, generator_kwargs):
    print()
    print()
    print(':'*100)
    print('RAG Model:')
    # init modules
    if retriever_kwargs != None:
        print(f"Retriever: {retriever_kwargs['init_args']['model_name']}")
    if reranker_kwargs != None:
        print(f"Reranker: {reranker_kwargs['init_args']['model_name']}")
    if generator_kwargs != None:
        print(f"Generator: {generator_kwargs['init_args']['model_name']}")

    print(':'*100)
    print()
    print()


def write_trec(fname, q_ids, d_ids, scores):
    with open(fname, 'w') as fout:
        for i, q_id in enumerate(q_ids):
            for rank, (d_id, score) in enumerate(zip(d_ids[i], scores[i])):
                fout.write(f'{q_id}\tq0\t{d_id}\t{rank+1}\t{score}\trun\n')


def write_generated(out_folder, out_filename, query_ids, questions, instructions, responses, labels, ranking_labels):
    jsonl_list = list()
    for i, (q_id, question, response, instruction, label, ranking_label) in enumerate(zip(query_ids, questions, responses, instructions, labels, ranking_labels)):
        jsonl = {}
        jsonl['q_id'] = q_id
        jsonl['response'] = response
        jsonl['instruction'] = instruction
        jsonl['label'] = label
        jsonl['question'] = question
        jsonl['ranking_label'] = ranking_label
        jsonl_list.append(jsonl)
    write_dict(out_folder, out_filename, jsonl_list)

def write_dict(out_folder, out_filename, dict_to_write):
    with open(f'{out_folder}/{out_filename}', 'w') as fp:
        json.dump(dict_to_write, fp, indent=2)

def load_trec(fname):
    # read file
    trec_dict = defaultdict(list)
    for l in tqdm(open(fname), desc=f'Loading existing trec run {fname}'):
        q_id, _, d_id, _, score, _ = l.split('\t')
        trec_dict[q_id].append((d_id, score))
    q_ids, d_ids, scores = list(), list(), list()
    for q_id in trec_dict:
        q_ids.append(q_id)
        d_ids_q, scores_q = list(), list()
        for d_id, score in trec_dict[q_id]:
            d_ids_q.append(d_id)
            scores_q.append(float(score))
        d_ids.append(d_ids_q)
        scores.append(scores_q)
    return q_ids, d_ids, scores



def eval_retrieval_kilt(experiment_folder, qrels_folder, query_dataset_name, doc_dataset_name, split, query_ids, doc_ids, scores, top_k=5, reranking=False, debug=False, write_trec=True):
    #only evaluate if wikipedia ids are in dataset
    # if all(sublist for sublist in doc_ids):
    #     return
    scores = scores.tolist() if torch.is_tensor(scores) else scores
    reranking_str = 're' if reranking else ''
    qrels_file = get_qrel_ranking_filename(qrels_folder, query_dataset_name, split, debug)
    if not os.path.exists(qrels_file): return
    qrel = json.load(open(qrels_file))
    if "doc_dataset_name" in qrel:
        if qrel["doc_dataset_name"] != doc_dataset_name: return
        qrel.pop("doc_dataset_name")
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'P_1', f'recall_{top_k}'})
    run = defaultdict(dict)
    for i, q_id in enumerate(query_ids):
        for i, (doc_id, score) in enumerate(zip(doc_ids[i], scores[i])):
            # if we have duplicate doc ids (because different passage can map to same wiki page) only write the max scoring passage
            if doc_id not in run[q_id]:
                run[q_id].update({doc_id: score}) 
            # if there is a higher scoring passage from the same wiki_doc, update the score (maxP)
            elif score >= run[q_id][doc_id]:
                run[q_id].update({doc_id: score}) 

    if write_trec:
        with open(f'{experiment_folder}/eval_{split}_{reranking_str}ranking_run.trec', 'w') as trec_out:
            for q_id, scores_dict in run.items():
                # Sort the dictionary by scores in decreasing order
                sorted_scores = dict(sorted(scores_dict.items(), key=lambda item: item[1], reverse=True))
                for i, (doc_id, score) in enumerate(sorted_scores.items()):
                    trec_out.write(f'{q_id}\tQO\t{doc_id}\t{i+1}\t{score}\trun\n')

    metrics_out = evaluator.evaluate(run)
    p_1 = sum([d["P_1"] for d in metrics_out.values()]) / max(1, len(metrics_out))
    recall = sum([d[f"recall_{top_k}"] for d in metrics_out.values()]) / max(1, len(metrics_out))
    
    mean_metrics = {'P_1':p_1, f'recall_{top_k}': recall}
    fname = f"eval_{split}_{reranking_str}ranking_metrics.json"
    write_dict(experiment_folder,  fname, mean_metrics)


def init_experiment(config, experiments_folder, index_folder, runs_folder, run_name, overwrite_exp=False, continue_batch=None):
    # if run_name != None hash self to get run_name to avoid overwriting and exp. folder mess
    run_name = f'tmp_{Hasher.hash(str(config))}' if run_name == None else f'tmp_{run_name}'
    experiment_folder = os.path.join(experiments_folder, run_name)
    print(f'Unfinished experiment_folder: {experiment_folder}')
    # get name of finished experiment
    finished_exp_name = get_finished_experiment_name(experiment_folder)
    if os.path.exists(finished_exp_name) and overwrite_exp:
        shutil.rmtree(finished_exp_name)
    # if experiment already exists finished quit
    if os.path.exists(finished_exp_name) and continue_batch == None:
        raise OSError(f"Experiment {finished_exp_name} already exists!")
    print('experiment_folder', finished_exp_name)

    # make dirs
    os.makedirs(experiments_folder, exist_ok=True)
    os.makedirs(index_folder, exist_ok=True)
    os.makedirs(runs_folder, exist_ok=True)
    os.makedirs(experiment_folder, exist_ok=True)
    # save entire config 
    OmegaConf.save(config=config, f=f"{experiment_folder}/config.yaml")
    # print config
    print(OmegaConf.to_yaml(config))


    wandb_project = f"NAVER-RAG-{experiments_folder.replace('/', '')}"
    os.environ["WANDB_PROJECT"] = wandb_project

    return run_name, experiment_folder


# IO

def get_finished_experiment_name(experiment_folder):
    return experiment_folder.replace('tmp_', '')

def move_finished_experiment(experiment_folder):
    shutil.move(experiment_folder, get_finished_experiment_name(experiment_folder))


def get_oracle_ranking_filename(runs_folder, dataset_name, split):
    return f'{runs_folder}/run.oracle.{dataset_name}.{split}.trec'

def get_qrel_ranking_filename(qrels_folder, dataset_name, split, debug=False):
    dataset_name = dataset_name.replace('_debug', '') if debug else dataset_name
    return f'{qrels_folder}/qrel.{dataset_name}.{split}.json'

def get_index_path(index_folder, dataset_name, model_name, query_or_doc, dataset_split='', query_generator_name='copy'):
    dataset_split = dataset_split + '_' if dataset_split != '' else ''
    query_gen_add = "" if query_generator_name == "copy" or query_or_doc=="doc" else f".{query_generator_name}"
    return os.path.join(index_folder,f'{dataset_name}_{dataset_split}{query_or_doc}_{model_name}{query_gen_add}')

def get_reranking_filename(runs_folder, query_dataset, doc_dataset, dataset_split, retriever_name, retrieve_top_k, reranker_name, rerank_top_k, query_generator_name):
    query_gen_add = "" if query_generator_name == "copy" else f".{query_generator_name}"
    return f'{runs_folder}/run.rerank.retriever.top_{retrieve_top_k}.{retriever_name}.rerank.top_{rerank_top_k}.{query_dataset}.{doc_dataset}.{dataset_split}.{reranker_name}{query_gen_add}.trec'

def get_ranking_filename(runs_folder, query_dataset, doc_dataset, retriever_name, dataset_split, retrieve_top_k, query_generator_name):
    if retriever_name == 'oracle_provenance':
        return get_oracle_ranking_filename(runs_folder, query_dataset, dataset_split)
    else:
        query_gen_add = "" if query_generator_name == "copy" else f".{query_generator_name}"
        return f'{runs_folder}/run.retrieve.top_{retrieve_top_k}.{query_dataset}.{doc_dataset}.{dataset_split}.{retriever_name}{query_gen_add}.trec'

def get_query_generation_filename(query_generation_folder, query_dataset, query_generator_name, split):
    return f'{query_generation_folder}/generated_queries.{query_dataset}.{split}.{query_generator_name}.json'
        
def get_embedding_datasets_path(embeddings_path):
    embeddings_path = embeddings_path.rstrip('/')
    return f'{embeddings_path}.hf'

def format_time(field_name, generation_time):
    return {field_name: time.strftime("%H:%M:%S.{}".format(str(generation_time % 1)[2:])[:11], time.gmtime(generation_time))}



def get_embeddings_dataset(embeddings_path, embedding_dim, num_proc=30):
    embeds_dataset_path = get_embedding_datasets_path(embeddings_path)
    if not os.path.exists(embeds_dataset_path):
        make_embeddings_dataset(embeddings_path, embedding_dim, num_proc)
    return datasets.load_from_disk(embeds_dataset_path)


def make_embeddings_dataset(embeddings_path, embedding_dim, num_proc):
    class StreamDatasetBuilder(datasets.GeneratorBasedBuilder):
        def _info(self):
            return datasets.DatasetInfo(
                description='dataset',
                features=datasets.Features(
                    {
                        "embedding":  datasets.Array2D(shape=(1, embedding_dim), dtype='float16'),
                    }
                ),
                supervised_keys=None,
                homepage="",
                citation='',
            )

        def _split_generators(self, dl_manager):
            emb_files = glob.glob(f'{embeddings_path}/*.pt')
            sorted_emb_files = sorted(emb_files, key=lambda x: int(''.join(filter(str.isdigit, x))))
            print(sorted_emb_files)
            return [
                datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": sorted_emb_files}),
            ]

        def _generate_examples(self, filepaths):
            id_ = 0
            for filepath in filepaths:
                embeds = torch.load(filepath)
                for emb in embeds:
                    yield id_, {'embedding': emb.unsqueeze(0)}
                    id_ += 1

    dataset_builder = StreamDatasetBuilder(config_name='Stream')
    dataset_builder.download_and_prepare(num_proc=num_proc)
    dataset = dataset_builder.as_dataset(split="train")
    dataset.save_to_disk(get_embedding_datasets_path(embeddings_path), num_proc=num_proc)


# adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py
def prepare_labels(input_ids, response_token_ids, ignore_index=-100):
    label_ids = input_ids.clone()

    for i in range(len(label_ids)):

        response_token_ids_start_idx = None

        for idx in np.where(label_ids[i] == response_token_ids[0])[0]:
            # `response_token_ids` is `'### Response:\n'`, here we are just making sure that the token IDs match
            if (
                response_token_ids
                == label_ids[i][idx : idx + len(response_token_ids)].tolist()
            ):
                response_token_ids_start_idx = idx

        if response_token_ids_start_idx is None:
            warnings.warn(
                f"This instance will be ignored in loss calculation. "
                f"Note, if this happens often, consider increasing the `max_seq_length`."
            )
            label_ids[i, :] = ignore_index
        else:
            response_token_ids_end_idx = response_token_ids_start_idx + len(response_token_ids)
            # Make pytorch loss function ignore all tokens up through the end of the response key
            label_ids[i, :response_token_ids_end_idx] = ignore_index
    return label_ids
