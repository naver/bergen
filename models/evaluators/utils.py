import numpy as np


def process_llm_outputs_assess_scores(outputs, options, unknown_value=-100):
    possible_scores = [[options[opt] for opt in options if opt in rep ] for rep in outputs]
    scores = [sc[0] if len(sc)==1 else unknown_value for sc in possible_scores]
    weird = [rep for i,rep in enumerate(outputs) if (len(possible_scores[i])==0 or len(possible_scores[i])>1)]
    return scores, weird


def get_mean_without_unknown(scores, unknown_value=-100):
    scores_to_consider = [s for s in scores if s!=unknown_value]
    if len(scores_to_consider)>0:
        return np.mean(scores_to_consider)
    else:
        return 0    
    

def unswitch_switched_scores(switched_scores: list, switches: list):
    """
    When we do pairwise comparison, we randomly switch the answer orders to prevent bias
    Here we de-switch the obtained scores
    """
    assert len(switched_scores) == len(switches), f"{len(switched_scores)} vs {len(switches)}"
    unswitched_scores = []
    for switched_score, switch in zip(switched_scores, switches):
        if not (0. <= switched_score <= 1.): # nothing we can do for weird scores
            unswitched_scores.append(switched_score)
        else:
            if switch:
                unswitched_scores.append(1 - switched_score)
            else:
                unswitched_scores.append(switched_score)
    return unswitched_scores


def get_pairwise_scores_without_unknown(scores, unknown_value=-100) -> dict:
    """
    Computes win/tie/lose scores for pairwise evaluation
    """
    valid_scores = [elt for elt in scores if 0. <= elt <= 1.]
    n_valid = max(1e-6, len(valid_scores)) # to avoid zero division
    return {
        'win': valid_scores.count(1)*100./n_valid,
        'tie': valid_scores.count(0.5)*100./n_valid,
        'lose': valid_scores.count(0)*100./n_valid
    }


def set_tq_description(tq, scores, weird, pairwise):
    """
    Utility to set tqdm description during evaluation, depending on pairwise vs pointwise.
    """
    if pairwise:
        tq.set_description(f"Win: {scores.count(1)*100./len(scores):4.1f}% tie {scores.count(0.5)*100./len(scores):4.1f}%\
            lose {scores.count(0)*100./len(scores):4.1f}%  weird {float(len(weird))/len(scores)*100:4.1f}%")
    else:
        tq.set_description(f" score: {get_mean_without_unknown(scores)* 100:4.1f}%, weird :{float(len(weird))/len(scores)*100:4.1f}%")
