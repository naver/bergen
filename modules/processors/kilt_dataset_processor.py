import datasets
import os
import urllib.request
import json
import pickle

from ..dataset_processor import Processor
from collections import defaultdict


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
        """
        response_files is a list of json files containing answers (TODO: make this cleaner.)
        """
        dataset_name = 'kilt_combined_qa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.response_files = response_files
        if response_files is not None:
            self.response_files = response_files
            self.use_cache = False # we'll use labels coming from some input file: we don't want to overwrite anything in this case.

    def process(self):
        dataset = datasets.load_dataset("dmrau/combined_qa")[self.split]
        if self.response_files is not None:
            new_data = {}
            # We load the files, they are output files from bergen runs.
            for response_file in self.response_files:
                print('Reading response file', response_file)
                with open(response_file, 'r') as f:
                    data = json.load(f)
                for elt in data:
                    new_data[elt['q_id']] = elt['response']

            # We assert that we obtained all ids:
            original_ids = set(dataset['id'])
            new_ids = set(new_data.keys())
            assert  original_ids == new_ids , f"{len(original_ids)} vs {len(new_ids)}"

            # Ans we replace
            def replace_label(example, idx):
                new_label = new_data[example['id']] # This is a str

                example['label'] = [new_label]
                return example

            dataset = dataset.map(replace_label, with_indices=True, desc='Replacing labels in dataset with read responses...')

        return dataset

    def get_dataset(self):
        print(f"Processing dataset {self.dataset_name} in {self.split} split ")
        debug_str = '_debug' if self.debug else ''
        assert self.dataset_name is not None # dataset name needs to be set in processor class
        # if self.dataset_name == 'kilt_combined_qa':
        #     print('Overrinding oracle for dataset loading ')
        #     oracle_provenance_str = ''
        # else:
        oracle_provenance_str = '_oracle_provenance' if self.oracle_provenance else ''
        # oracle_provenance_str = ''
        out_folder = os.path.join(f'{self.out_folder}', f'{self.dataset_name}_{self.split}{oracle_provenance_str}')
        if os.path.exists(out_folder) and not self.overwrite and self.use_cache:
            dataset = datasets.load_from_disk(out_folder)
            if self.debug:
                dataset = dataset.select(range(min(50, len(dataset))))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)
            #id2index = self.tsv_to_dict(f'{out_folder}/id2index.csv')
            id2index = pickle.load(open(f'{out_folder}/id2index.p', 'rb'))
        else:
            dataset = self.process()
            id2index = self.get_index_to_id(dataset) 
            if self.use_cache: # saving only if use_cache set (true most of the time)
                dataset.save_to_disk(out_folder)
                pickle.dump(id2index, open(f'{out_folder}/id2index.p', 'wb'))
            if self.debug:
                dataset = dataset.select(range(15))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)
                
        dataset.id2index = id2index
        dataset.name = self.dataset_name + debug_str + oracle_provenance_str

        return dataset


class KiltMultiQAMSMarco(Processor):
    """
    Dataset combining multi QA and MS MArco data, mostly used for FT of OSCAR models.
    """
    def __init__(self,  *args, **kwargs):
        dataset_name = 'kilt_combined_qa_ms_marco'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        # No response file: it's purely Solar here.

    def process(self):
        return datasets.load_from_disk("/scratch/1/user/mlouis/calmar/data/kilt_combined_qa_ms_marco")
    
    def get_dataset(self):
        print(f"Processing dataset {self.dataset_name} in {self.split} split ")
        debug_str = '_debug' if self.debug else ''
        assert self.dataset_name is not None # dataset name needs to be set in processor class
        # if self.dataset_name == 'kilt_combined_qa':
        #     print('Overrinding oracle for dataset loading ')
        #     oracle_provenance_str = ''
        # else:
        oracle_provenance_str = '_oracle_provenance' if self.oracle_provenance else ''
        # oracle_provenance_str = ''
        out_folder = os.path.join(f'{self.out_folder}', f'{self.dataset_name}_{self.split}{oracle_provenance_str}')
        if os.path.exists(out_folder) and not self.overwrite and self.use_cache:
            dataset = datasets.load_from_disk(out_folder)
            if self.debug:
                dataset = dataset.select(range(min(50, len(dataset))))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)
            #id2index = self.tsv_to_dict(f'{out_folder}/id2index.csv')
            id2index = pickle.load(open(f'{out_folder}/id2index.p', 'rb'))
        else:
            dataset = self.process()
            id2index = self.get_index_to_id(dataset) 
            if self.use_cache: # saving only if use_cache set (true most of the time)
                dataset.save_to_disk(out_folder)
                pickle.dump(id2index, open(f'{out_folder}/id2index.p', 'wb'))
            if self.debug:
                dataset = dataset.select(range(15))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)

        dataset.id2index = id2index
        dataset.name = self.dataset_name + debug_str + oracle_provenance_str

        return dataset
