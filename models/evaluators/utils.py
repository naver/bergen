import numpy as np


def process_llm_outputs_assess_scores(outputs, options, unknown_value=-100):
    
    possible_scores = [[options[opt] for opt in options if opt.lower() in rep.lower() ] for rep in outputs]
    scores = [sc[0] if len(sc)==1 else unknown_value for sc in possible_scores]
    weird = [rep for i,rep in enumerate(outputs) if (len(possible_scores[i])==0 or len(possible_scores[i])>1)]
    return scores, weird

def get_mean_without_unknown(scores, unknown_value=-100):
    scores_to_consider = [s for s in scores if s!=unknown_value]
    if len(scores_to_consider)>0:
        return np.mean(scores_to_consider)
    else:
        return 0    

