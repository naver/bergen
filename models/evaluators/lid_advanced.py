import fasttext
import langid
from ftlangdetect import detect
from huggingface_hub import hf_hub_download

class LID_advanced:
    """
    This is the advanced version of LID (different from the one used to compute results at https://aclanthology.org/2024.knowllm-1.15/)
    - it ensembles output of multiple LID models for more reliable results
    - takes into account the results on the gold answer. 
    It allows to take into detect the examples of Named Entities that would have the same form across languages, and often detected as English.
    - 
    """
    def __init__(self, target_lang):
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.target_lang = target_lang 
    
    def __call__(self, predictions, references, questions):
        def correct_lang(response, gold, question, lang):
            response = response.replace("\n", " ")
            #compute response of fasttext lid
            response_l = detect(text=response, low_memory=False)['lang']
            if response_l == lang:
                return 1
            
            #compute response by langid
            lid_response_l = langid.classify(response)[0]
            if lid_response_l == lang:
                return 1
            # if none of responses match expected target language
            # compute language of gold label with fasttext and langid and check whether it maps expected target language
            # 
            gold_l = set([detect(text=g, low_memory=False)['lang'] for g in gold])
            lid_gold = set([langid.classify(g)[0] for g in gold])
            gold_l = lid_gold.union(gold_l)
            response = response.lower().replace(".", "").strip()
            gold = [g.lower() for g in gold]
            if response in gold: # if response matches one of the gold responses, assume lid is correct
                return 1
            if len(response)>20: #if lid didn't match for long enough responses: it is probably wrong language
                #print(f"WRONG, {lang}: {response} ({response_l, lid_response_l}) vs {gold}, {gold_l}")
                return 0  
            if response_l in  gold_l or lid_response_l in gold_l: # for short responses, if its lid matches glod lid: assume it is correct (it would be mostly en, but could be smth else in case of foreign Named Entities)
                #print(f"CORRECT, {lang}: {response} ({response_l, lid_response_l}) vs {gold}, {gold_l}")
                return 1 
            #print("Still not covered", response, response_l, lid_response_l, len(response), gold, gold_l, lid_gold, lang)
            #print(f"WRONG, {lang}: {response} ({response_l, lid_response_l}) vs {gold}, {gold_l}")
            #the case when neither response nor gold label match target lng and shorter than 20 characters, exclude it from computation, could be case of person names)
            return -1 
        scores = []
        correct = 0
        denom = 0
        skip = 0
        for i, response in enumerate(predictions):
            iscorrect = correct_lang(response, references[i], questions[i], self.target_lang)
            if iscorrect == -1:
                skip +=1
            scores.append(iscorrect)
            correct += iscorrect
            denom += 1    
        print("Percentage of skipped samples ", float(skip)/i)
        return correct/denom, scores
