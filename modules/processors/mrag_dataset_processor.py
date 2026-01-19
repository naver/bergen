from ..dataset_processor import Processor
import datasets
from datasets import Dataset
import os
import numpy as np
import pandas as pd
import random

class MKQA(Processor):
    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'mkqa_{lang}'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang
        
    def process(self):
        mkqa = datasets.load_dataset('mkqa', trust_remote_code=True)
        kilt_nq = datasets.load_dataset("kilt_tasks", "nq")

        mkqa_ids = {s['example_id']:i for i, s in enumerate(mkqa[self.split])}
        kilt_nq_train_ids = {s['id']:i for i, s in enumerate(kilt_nq[self.split])}

        overlap_ids = set(mkqa_ids.keys()).intersection(set(kilt_nq_train_ids.keys()))
        overlap_mkqa = mkqa['train'].select([mkqa_ids[i] for i in overlap_ids])
        overlap_kilt_nq = kilt_nq['train'].select([kilt_nq_train_ids[i] for i in overlap_ids])        
        dataset = overlap_kilt_nq.add_column(f"content", [sample['queries'][self.lang] for sample in overlap_mkqa])    
        # discarding empty answers
        dataset = dataset.add_column(f"label", [[a['text'] for a in sample['answers'][self.lang] if not a['text']==None] for sample in overlap_mkqa])
        # filter out samples with empty answer
        dataset = dataset.filter(lambda example: len(example['label'])>0)

        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el[f'answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})        
        dataset = dataset.remove_columns(['meta'])
        return dataset


class XORQA(Processor):
    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'xor_tydiqa_{lang}'
        self.lang = lang
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def extend(label, lang):
            if "yes" in label:
                label += {"ru": ["да"], "ko": ["예"], "ja": ["はい"], "fi": ["kyllä", "joo"], "ar": ["نعم", "أجل", "بلى"]}[lang]
            if "no" in label:
                label += {"ru": ["нет"], "ko": ["아니요"], "ja": ["いいえ"], "fi": ["ei"], "ar": ["لا"]}[lang]
            return label
        os.system("wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl")
        dataset = datasets.load_dataset("json", data_files="xor_dev_full_v1_1.jsonl")["train"] # the file should be already .dev, and train is just default hf label
        dataset = dataset.filter(lambda example: example['lang']==self.lang)
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': extend([el for el in example['answers'] if len(el) > 0], self.lang)})
        dataset = dataset.rename_column("question", "content")
        dataset = dataset.map(lambda x: {'id': str(x['id'])}) # ids should be strings.
        #dataset = dataset.remove_columns(['meta', 'output'])
        os.system("rm xor_dev_full_v1_1.jsonl")
        return dataset


class TydiQA(Processor):
    def __init__(self, langcode="en", language="english", *args, **kwargs):
        dataset_name = f'tydiqa_{langcode}'
        self.language = language
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        dataset =  datasets.load_dataset("google-research-datasets/tydiqa", "secondary_task")[self.split] 
        dataset = dataset.filter(lambda example: example['id'].startswith(self.language))
        dataset = dataset.map(lambda example: {'label': [el for el in example['answers']["text"] if len(el) > 0]})
        dataset = dataset.rename_column("question", "content")
        dataset = dataset.remove_columns(['title', 'context', 'answers'])
        return dataset

class MIRACL(Processor):

    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'miracl_{lang}'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang
        
    def process(self):
        dataset = datasets.load_dataset('miracl/miracl', self.lang)[self.split]
        dataset = dataset.rename_column("query_id", "id") 
        dataset = dataset.rename_column("query", "content") 
        dataset = dataset.map(lambda example: {'label': [ps['text'] for ps in example['positive_passages'] if len(ps) > 0]})
        dataset = dataset.remove_columns(['positive_passages', 'negative_passages'])
        return dataset

class XPQA(Processor):
    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'xpqarand_{lang}'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang
        
    def process(self):
        df_raw = pd.read_csv("https://raw.githubusercontent.com/amazon-science/contextual-product-qa/refs/heads/main/xPQA/test_answerable_corrected.csv")
        df = df_raw[df_raw['lang']==self.lang]

        gdf = df.groupby('qid').agg({
            'question': 'first',
            'answer': lambda x: list(x),
            'context': lambda x: list(x),
        }).reset_index()
        def process_data(row):
            nan_context = []
            gcontext = []
            for a, c in zip(row['answer'], row['context']):
                if a is np.nan:
                    nan_context.append(c)
                else:
                    gcontext.append((c, a))
            return {
                'qid': row['qid'],
                'question': row['question'],
                'context': gcontext,
                'nan_context': nan_context
            }
        gdf = pd.DataFrame(gdf.apply(process_data, axis=1).tolist())
        gdf = gdf.explode('context').reset_index(drop=True)
        gdf['answer'] = gdf['context'].apply(lambda x: [x[1]])
        def create_context(row):
            all_context = row['nan_context'] + [row['context'][0]] 
            random.shuffle(all_context)
            return ['\n'.join(all_context)]
        gdf['context'] = gdf.apply(lambda x: create_context(x), axis=1)
        gdf = gdf.drop(columns=['nan_context'])
        gdf['qid'] = gdf.index.astype(str)

        gdf = gdf[gdf['answer'].map(lambda x: (len(x) > 0) and (len(str(x[0])) > 0))]
        gdf.rename(columns={'qid': 'id', 'question': 'content', 'answer': 'label', 'context': 'doc'}, inplace=True)
        # convert id to str
        gdf['id'] = gdf['id'].astype(str)
        gdf['doc_id'] = gdf['id']
        dataset = Dataset.from_pandas(gdf)
        return dataset

class MedExpQAExp(Processor):
    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'medexpqaexp_{lang}'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang
        
    def process(self):
        hf_name = 'HiTZ/MedExpQA' 
        dataset = datasets.load_dataset(hf_name, self.lang, num_proc=self.num_proc)[self.split]
        def make_question(x):
            choices_str = '\n\n'.join([f"{i}. {c}" for i, c in x['options'].items()])
            if self.lang == 'en':
                return {'content': f"{x['full_question']}\nHere are the potential choices:\n\n{choices_str}\n\nThe correct answer is:"}
            elif self.lang == 'es':
                return {'content': f"{x['full_question']}\nAquí están las posibles opciones:\n\n{choices_str}\n\nLa opción correcta es:"}
            elif self.lang == 'fr':
                return {'content': f"{x['full_question']}\nVoici les options possibles:\n\n{choices_str}\n\nLa bonne option est:"}
            elif self.lang == 'it':
                return {'content': f"{x['full_question']}\nEcco le opzioni possibili:\n\n{choices_str}\n\nL'opzione corretta è:"}
        dataset = dataset.map(make_question, num_proc=self.num_proc, batched=False)


        def get_right_answer(example):
            answer_index = example['correct_option']
            return example['options'][str(answer_index)]

        dataset = dataset.map(lambda example, idx: {'id': str(idx), 
                                                    "label": [str(example["correct_option"]) + '. ' + get_right_answer(example)],
                                                    "doc": [example['full_answer']],
                                                    "doc_id": str(idx)}, with_indices=True)
        dataset = dataset.remove_columns(['correct_option', 'explanations', 'full_answer', 
                                          'full_answer_no_ref', 'full_question', 'lang', 'options', 
                                          'question_id_specific', 'type', 'year', 'rag'])
        dataset = dataset.filter(lambda example: len(example['label'])>0)
        dataset = dataset.filter(lambda example: len(example['content'])>0)
        return dataset
