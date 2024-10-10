'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
from collections import defaultdict
from hydra.utils import instantiate
# reranking
class Rerank():
    def __init__(self, init_args=None, batch_size=1):

        self.batch_size = batch_size
        self.init_args = init_args
        self.model = instantiate(self.init_args)
    @torch.no_grad()
    def eval(self, dataset):
        # get dataloader
        self.model.model.to('cuda')
        dataloader = DataLoader(dataset, batch_size=self.batch_size, collate_fn=self.model.collate_fn)
        q_ids, d_ids, scores, embs_list = list(), list(), list(), list()
        # run inference on the dataset
        for batch in tqdm(dataloader, desc=f'Reranking: {self.model.model_name}'):
            q_ids += batch.pop('q_id')
            d_ids += batch.pop('d_id')
            outputs = self.model(batch)
            score = outputs['score']
            scores.append(score)
            
        # get flat tensor of scores        
        scores = torch.cat(scores).ravel()
        # sort by scores 
        q_ids_sorted, d_ids_sorted, scores_sorted = self.sort_by_score_indexes(scores, q_ids, d_ids)
        self.model.model.to('cpu')
        torch.cuda.empty_cache()
        return {
            "score": scores_sorted,
            "doc_id": d_ids_sorted,
            "q_id": q_ids_sorted,
            }
    
   # takes scores (num_queries x top_k) and q_ids and sortes them per q_id by score and returns list of sorted idxs
    def sort_by_score_indexes(self, scores, q_ids, d_ids):
        ranking = defaultdict(list)
        q_ids_sorted, doc_ids_sorted, scores_sorted = list(), list(), list()
        for i, (q_id, d_id) in enumerate(zip(q_ids, d_ids)):
            ranking[q_id].append((scores[i], d_id))
        for q_id in ranking:
            # sort (score, doc_id) by score
            sorted_list = sorted(ranking[q_id], key=lambda x: x[0], reverse=True)
            score_sorted, d_id_sorted = zip(*sorted_list)
            scores_sorted.append(torch.stack(score_sorted))
            doc_ids_sorted.append(list(d_id_sorted))
            q_ids_sorted.append(q_id)
        # this conversion to tensor is useless and breaks for queries that return less than top_k documents
        # so keeping them as a list
        #scores_sorted = torch.stack(scores_sorted)
        return q_ids_sorted, doc_ids_sorted, scores_sorted

    def get_clean_model_name(self):
        return self.model.model_name.replace('/', '_')