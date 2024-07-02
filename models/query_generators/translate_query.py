from models.query_generators.query_generator import QueryGenerator
from typing import List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
from modules.metrics import normalize
import torch

class TranslateQuery(QueryGenerator):
    def __init__(self, model_name, src_lang, tgt_lang, normalize_translations=True, translation_max_length=100, batch_size=1):
        self.name = f"translated_query_{src_lang}_{tgt_lang}"
        self.model_name = model_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.batch_size = batch_size
        self.normalize_translations = normalize_translations
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.translator = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.translation_pipeline = pipeline('translation', 
                                model=self.translator, 
                                tokenizer=self.tokenizer, 
                                src_lang=self.src_lang, 
                                tgt_lang=self.tgt_lang, 
                                max_length=translation_max_length,
                                device=self.device)
        #self.translator.half() # fp16
        self.translator.eval()
        
    def generate(self, user_questions: List[str]):
        num_q = len(user_questions)
        num_batches = (num_q - 1) // self.batch_size + 1
        queries = []
        for batch_idx in tqdm(range(num_batches), desc=f'Translating queries...', total=num_batches):
            batch = user_questions[batch_idx*self.batch_size:(batch_idx+1)*self.batch_size]
            translations = self.translation_pipeline(batch)
            translations = [tr['translation_text'] for tr in translations]
            if self.normalize_translations: 
                translations = [normalize(t) for t in translations]
            queries += translations
        return queries