from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef
import string
import regex 
import numpy as np
from rouge import Rouge
from tqdm import tqdm
from collections import Counter

# partly adapted from https://github.com/facebookresearch/atlas/blob/0ec8889492d5187b26c51b8d1781239a4cf6741e/src/evaluation.py

rouge = Rouge()


def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = float(f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
        "f1": f1,
    }

def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

def normalize(s: str) -> str:
    def remove_articles(text):
        return regex.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_single(prediction, ground_truth, tokenfun=lambda x: x.split()):
    prediction_tokens = tokenfun(prediction)
    ground_truth_tokens = tokenfun(ground_truth)
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def ngrams(s, n=3):
    exclude = set(string.punctuation)
    s = ''.join(ch if ch not in exclude else " " for ch in s.lower())
    tokens = []
    for w in s.split():
        l = len(w)
        if l < n:
            tokens.append(w)
        else:
            for i in range(l-n+1):
                tokens.append(w[i:i+n])
    return tokens

def rouge_wrapper(prediction, ground_truth):
    try:
        result = rouge.get_scores(prediction, ground_truth, avg=True)
        return result["rouge-1"]["f"], result["rouge-2"]["f"], result["rouge-l"]["f"]
    except:
        return 0.0, 0.0, 0.0


def rouge_score_single(prediction, ground_truths):
    ground_truths = [x for x in ground_truths if len(x) > 0]
    if len(prediction) == 0 or len(ground_truths) == 0:  
        # check if empty prediction or if there is no hypothesis with len > 0
        return 0.0, 0.0, 0.0
    scores = [rouge_wrapper(prediction, gt) for gt in ground_truths]
    rouge1 = max(s[0] for s in scores)
    rouge2 = max(s[1] for s in scores)
    rougel = max(s[2] for s in scores)
    return rouge1, rouge2, rougel

def rouge_score(predictions, references):
    rouge1, rouge2, rougel = list(), list(), list()
    for ground_truths, predicition in zip(references, predictions):
        rouge1_, rouge2_, rougel_ = rouge_score_single(predicition, ground_truths) 
        rouge1.append(rouge1_)
        rouge2.append(rouge2_)
        rougel.append(rougel_)
    return np.mean(rouge1), np.mean(rouge2), np.mean(rougel)


def f1_score(predictions, references, tokenfun=lambda x: x.split()):
    f1, precision, recall = list(), list(), list()
    for ground_truths, prediction in zip(references, predictions):
        f1_, precision_, recall_ = [max(values) for values in zip(*[f1_single(prediction, gt, tokenfun) for gt in ground_truths])]
        f1.append(f1_)
        precision.append(precision_)
        recall.append(recall_)
    return np.mean(f1), np.mean(precision), np.mean(recall)

def em_single(prediction, ground_truth):
    return float(normalize(prediction) == normalize(ground_truth))


def exact_match_score(predictions, references):
    return np.mean([max([em_single(prediction, gt) for gt in ground_truths]) for ground_truths, prediction in zip(references, predictions)])

def match_single(prediction, ground_truth):
    return float(normalize(ground_truth) in normalize(prediction))


def match_score(predictions, references):
    return np.mean([max([match_single(prediction, gt) for gt in ground_truths]) for ground_truths, prediction in zip(references, predictions)])



class RAGMetrics:
    @staticmethod
    def compute(predictions, references, questions=None):
        rouge1, rouge2, rougel = rouge_score(predictions, references)
        f1, precision, recall = f1_score(predictions, references)
        f1_char3gram, precision_char3gram, recall_char3gram = f1_score(predictions, references, ngrams)
        return {    "M": match_score(predictions, references),
                    "EM": exact_match_score(predictions, references),
                    #"BEM": self.bem(predictions, references, questions),
                    "F1": f1,
                    "Precision": precision, 
                    "Recall": recall,
                    "Recall_char3gram": recall_char3gram,
                    "Rouge-1": rouge1,
                    "Rouge-2": rouge2,
                    "Rouge-L": rougel,
                }

