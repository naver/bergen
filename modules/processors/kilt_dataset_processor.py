from ..dataset_processor import *
import datasets
import os
from collections import defaultdict
import urllib.request
import json

class KILTNQ(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_nq'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "nq"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})

        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset


class KILTTriviaqa(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_triviaqa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        
        # Get the KILT task datasets
        dataset = datasets.load_dataset("kilt_tasks", name="triviaqa_support_only")[self.split]
        hf_q_ids = set(dataset['id'])
        # Most tasks in KILT already have all required data, but KILT-TriviaQA
        # only provides the question IDs, not the questions themselves.
        # Thankfully, we can get the original TriviaQA data with:
        trivia_qa = datasets.load_dataset('trivia_qa', 'unfiltered.nocontext')[self.split]
        # The KILT IDs can then be mapped to the TriviaQA questions with:

        def add_missing_data(x, trivia_qa_subset, triviaqa_map):
            i = triviaqa_map[x['id']]
            x['input'] = trivia_qa_subset[i]['question']
            x['output'][0]['original_answer'] = trivia_qa_subset[i]['answer']['value']
            return x
            
        triviaqa_map = dict([(q_id, i) for i, q_id in enumerate(trivia_qa['question_id'])])
        dataset = dataset.filter(lambda x: x['id'] in triviaqa_map, num_proc=self.num_proc)
        # only use ids that are present in the kilt_dataset
        dataset = dataset.filter(lambda x: x['id'] in hf_q_ids, num_proc=self.num_proc)
        dataset = dataset.map(add_missing_data, fn_kwargs=dict(trivia_qa_subset=trivia_qa, triviaqa_map=triviaqa_map), num_proc=self.num_proc)

        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset

class KILTHotpotqa(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_hotpotqa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "hotpotqa"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset
    
class KILTAidayago2(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_aidayago2'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "aidayago2"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset
    
class KILTCweb(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_cweb'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "cweb"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset
    
class KILTEli5(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_eli5'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)


    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "eli5"

        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]

        # hf dataset is missing provenances add them
        if self.split == 'dev':
            original_fname = '../data/eli5-dev-kilt.jsonl'
            if not os.path.exists(original_fname):
                urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl", original_fname)

            original = defaultdict(dict)
            for l in open(original_fname):
                example = json.loads(l)
                original_provenance = [out['provenance'] for out in example['output'] if 'provenance' in out]
                q_id = example['id']
                original[q_id] = original_provenance

            def add_missing_data(x, original_dataset):
                id_ = x['id']
                original_provenance = original_dataset[id_]
                for i in range(len(original_provenance)):
                    x['output'][i]['provenance'] = original_provenance[i]
                return x
            
            dataset = dataset.map(add_missing_data, fn_kwargs=dict(original_dataset=original))
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset
    
class KILTFever(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_fever'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "fever"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset
    
class KILTStructuredZeroshot(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_structured_zeroshot'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "structured_zeroshot"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset


class KILTSTrex(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_trex'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "trex"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset
    
class KILTWned(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_wned'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "wned"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset

class KILTWow(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt_wow'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_tasks' 
        hf_query_or_doc_name= "wow"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el['answer'] for el in example['output'] if len(el['answer']) > 0]})
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['meta', 'output'])
        return dataset


class KILTMULTIQA(Processor):
    def __init__(self, response_files: list = None, *args, **kwargs):
        dataset_name = 'kilt_combined_qa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
            
    def process(self):
        return datasets.load_dataset("dmrau/combined_qa")[self.split]
        