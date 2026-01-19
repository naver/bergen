import pytrec_eval
from collections import defaultdict
from tqdm import tqdm

def load_trec_dict(fname):
    # read file
    trec_dict = defaultdict(dict)
    for l in tqdm(open(fname), desc=f'Loading existing trec run {fname}'):
        q_id, _, d_id, _, score, _ = l.split('\t')
        trec_dict[q_id][d_id] = float(score)
    return trec_dict

def evaluate_retrieval_simple(run, qrel, metrics):
    evaluator = pytrec_eval.RelevanceEvaluator(qrel, metrics)
    return evaluator.evaluate(run)