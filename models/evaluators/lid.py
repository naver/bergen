import fasttext
import langid
from ftlangdetect import detect
from huggingface_hub import hf_hub_download
from langcodes import *

class LID:
    """
    This is the basic version of LID used to compute results at https://aclanthology.org/2024.knowllm-1.15/
    """
    def __init__(self, target_lang):
        model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model = fasttext.load_model(model_path)
        self.target_lang = target_lang # Flores lang codes: 
        # https://github.com/facebookresearch/flores/blob/main/flores200/README.md
    
    def __call__(self, predictions, _, __, instructions=None):
        def correct_lang(response):
            response = response.replace("\n", " ")
            return int(Language.get(self.model.predict(response)[0][0].removeprefix('__label__')).language == self.target_lang)
        scores = []
        correct = 0
        denom = 0
        skip = 0
        for response in predictions:
            if len(response) > 20:
                iscorrect = correct_lang(response)
                scores.append(iscorrect)
                correct += iscorrect
                denom += 1
            else:
                scores.append(-1)
                skip +=1
        print("Percentage of skipped samples ", float(skip)/len(predictions))
        return correct/denom, scores
   
    """
    def __call__(self, predictions, references, questions, instructions=None):
        def correct_lang(response, gold, question, lang):
            #langid.set_languages([lang, 'en'])
            response = response.replace("\n", " ")
            response_l = detect(text=response, low_memory=False)['lang']
            if response_l == lang:
                return 1
            
            lid_response_l = langid.classify(response)[0]
            if lid_response_l == lang:
                return 1
            gold_l = set([detect(text=g, low_memory=False)['lang'] for g in gold])
            lid_gold = set([langid.classify(g)[0] for g in gold])
            gold_l = lid_gold.union(gold_l)
            response = response.lower().replace(".", "").strip()
            gold = [g.lower() for g in gold]
            if response in gold:
                return 1
            if len(response)>20:
                #print(f"WRONG, {lang}: {response} ({response_l, lid_response_l}) vs {gold}, {gold_l}")
                return 0  
            if response_l in  gold_l or lid_response_l in gold_l:
                #print(f"CORRECT, {lang}: {response} ({response_l, lid_response_l}) vs {gold}, {gold_l}")
                return 1 
            #print("Still not covered", response, response_l, lid_response_l, len(response), gold, gold_l, lid_gold, lang)
            #print(f"WRONG, {lang}: {response} ({response_l, lid_response_l}) vs {gold}, {gold_l}")
            return 0 #the case when neither response nor gold label match target lng (could be case of person names)
        scores = []
        correct = 0
        denom = 0
        skip = 0
        for i, response in enumerate(predictions):
            iscorrect = correct_lang(response, references[i], questions[i], self.target_lang)
            scores.append(iscorrect)
            if iscorrect == -1:
                skip +=1
                continue
            correct += iscorrect
            denom += 1    
        #print(skip)
        return correct/denom, scores
    """
    