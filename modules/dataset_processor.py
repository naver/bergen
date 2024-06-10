'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import datasets
import os
from collections import defaultdict
import csv
from tqdm import tqdm
import pickle
from hydra.utils import instantiate
import urllib.request
import json
import requests  
from functools import partial


# Base class that every processor interhits from 
class Processor(object):
    
    def __init__(self, 
                dataset_name,
                split, 
                out_folder, 
                num_proc, 
                overwrite, 
                debug, 
                oracle_provenance, 
                shuffle_labels
                ):
        self.dataset_name = dataset_name
        self.split = split
        self.num_proc = num_proc
        self.out_folder = out_folder
        self.overwrite = overwrite
        self.debug = debug
        self.oracle_provenance = oracle_provenance
        self.shuffle_labels = shuffle_labels

    def process():
        raise NotImplementedError()
    
    def add_index(self, dataset):
        dataset = dataset.add_column("index", range(len(dataset)))    
        return dataset
    
    def get_index_to_id(self, dataset):
        if 'index' not in dataset.features:
            dataset = self.add_index(dataset)
        return dict(zip(dataset["id"], dataset["index"]))
    
    def shuffled_labels_as_content(self, dataset):
        import random
        random.seed(42)
        col = dataset['label']
        random.shuffle(col)
        dataset_dict = dataset.to_dict()
        dataset_dict['ranking_label'] = [el[0] for el in col]
        return datasets.Dataset.from_dict(dataset_dict)

    def get_dataset(self):
        print(f"Processing dataset {self.dataset_name} in {self.split} split ")
        debug_str = '_debug' if self.debug else ''
        assert self.dataset_name != None # dataset name needs to be set in processor class
        oracle_provenance_str = '_oracle_provenance' if self.oracle_provenance else ''
        out_folder = os.path.join(f'{self.out_folder}', f'{self.dataset_name}_{self.split}{oracle_provenance_str}')
        if os.path.exists(out_folder) and not self.overwrite:
            dataset = datasets.load_from_disk(out_folder)
            if self.debug:
                dataset = dataset.select(range(15))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)
            #id2index = self.tsv_to_dict(f'{out_folder}/id2index.csv')
            id2index = pickle.load(open(f'{out_folder}/id2index.p', 'rb'))
            dataset.id2index = id2index
        else:
            dataset = self.process()
            dataset.save_to_disk(out_folder)
            id2index = self.get_index_to_id(dataset) 
            pickle.dump(id2index, open(f'{out_folder}/id2index.p', 'wb'))
            if self.debug:
                dataset = dataset.select(range(15))
            if self.shuffle_labels:
                dataset = self.shuffled_labels_as_content(dataset)
            dataset.id2index = id2index
            #self.dict_to_tsv(id2index, f'{out_folder}/id2index.csv')
        dataset.name = self.dataset_name + debug_str + oracle_provenance_str
        return dataset
    
    def dict_to_tsv(self, id_to_index, file_path):
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                # Write data
                for id, index in id_to_index.items():
                    row = f"{id}\t{index}\n"
                    file.write(row)
        except Exception as e:
            print(f"Error writing id2index file: {e}")

    def tsv_to_dict(self, file_path):
        try:
            id_to_index = {}
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file, delimiter='\t')
                for row in reader:
                    if len(row) == 2:
                        id, index = row
                        id_to_index[id] = int(index)
            return id_to_index
        except Exception as e:
            print(f"Error loading id2index file: {e}")
            return None

# ---------------------------------------- #
# query processors
# ---------------------------------------- #
        
class BIOASQ11B(Processor):

    def __init__(self, data_path, *args, **kwargs):
        self.dataset_name = 'BIOASQ11B'
        self.path = data_path
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        dataset = datasets.load_dataset("csv", data_files=[self.path], delimiter="\t", )[self.split]
        #['id','docs','question','type','ideal_answer','exact_answer','snippets']
        dataset = dataset.map(lambda example: {'label': eval(example['ideal_answer'])})
        dataset = dataset.rename_column("question", "content")
        dataset = dataset.remove_columns(['docs', 'type','exact_answer','snippets'])
        return dataset
    
    


class WIKIQA(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'wiki_qa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'wiki_qa' 
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        # discarding empty answers 
        dataset_f = dataset.filter (lambda x: x['label'] == 1) # keeping only the valid sentences
        
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        # No ranking labels
        #dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})

        dataset_l=[]

        qid_set=set(dataset_f['question_id'])

        for q in qid_set:
            qsel= dataset_f.filter(lambda x: x['question_id']==q)
            ex={ 'id':q ,
                 'content':qsel['question'][0],
                  'label': qsel['answer']}
            dataset_l.append(ex)
        
        #dataset = dataset_f.rename_column("question", "content")
        #dataset = dataset.remove_columns(['document_title', 'label'])
        #dataset = dataset.rename_column("answer", "label")
        #dataset = dataset.rename_column("question_id", "id")
        dataset=datasets.Dataset.from_list(dataset_l)
        return dataset

class SCIQ(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'sciq'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'sciq' 
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("question", "content")
        #dataset = dataset.rename_column("correct_answer", "label")
        dataset = dataset.map(lambda example: {'label': [example['correct_answer']]})
        dataset = dataset.remove_columns(["correct_answer","distractor1", "distractor2","distractor3","support"])
        #generating the id, train_0 ... validation_0 validation_1
        cid= [self.split+str(i) for i in range(len(dataset))]
        dataset = dataset.add_column("id", cid)

        return dataset




class ASQA(Processor):
    wiki_api = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles={}"
    def __init__(self, *args, **kwargs):
        dataset_name = 'asqa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    @staticmethod
    def fetch_wiki_id(inp):
        wiki_link, wiki_title = inp
        response = requests.get(ASQA.wiki_api.format(wiki_title))
        wiki_object_has_no_pages = 0
        if response.status_code == 200:
            data = response.json()
            # hack because of weird dict structure returned json of the wiki_api
            pages = data.get("query", {}).get("pages", {})
            if pages:
                wiki_id = next(iter(pages.keys()))
                #return (wiki_link, wiki_id)
                return wiki_id
            else:
                wiki_object_has_no_pages += 1
                return None
        else:
            count_not_found += 1
            print(f"wiki page {wiki_link} could not be fetched from wiki!")
            return None


    def process(self):

        hf_name = 'din0s/asqa' 
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        
        #features: ['ambiguous_question', 'qa_pairs', 'wikipages', 'annotations', 'sample_id'],

        # use "sample_id" ?
        dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)

        #dataset = dataset.rename_column("answer", "label")
        dataset = dataset.rename_column("ambiguous_question", "content")

        # get short answers
        def short_answers(example):
            #z=[ q for ex in dataset['dev'][:2]['qa_pairs']  for x in ex for q in x['short_answers']]~
            z= list(set([ ans for qa in example['qa_pairs']  for ans in qa['short_answers'] ]))
            # or z=[ x['short_answers']  for ex in qa_pairs for x in ex]
            return z #[pair[] for pair in qa_pairs]
        def get_wiki_id(example):
                wiki_ids=list()
                wiki_objects = example['wikipages']
                for wiki_object in wiki_objects:
                        if wiki_object['url'] != None:
                            # get wiki url
                            wiki_link = wiki_object['url']
                            # get title
                            wiki_title = wiki_link.split("/")[-1]
                            # fetch id by title
                            wiki_ids.append(ASQA.fetch_wiki_id((wiki_link, wiki_title)))
                return wiki_ids


        # long_awser ?
        #  example['annotation'][0]['long_answer']
        # Apply the cleaning function to the 'label' column
        dataset = dataset.map(lambda example: {'label': short_answers(example)})
        #dataset = dataset.map(lambda example: {'ranking_label': get_wiki_id(example)},num_proc=5)

        dataset = dataset.remove_columns([ 'qa_pairs', 'wikipages', 'annotations', 'sample_id'])

        # ranking_label: wikipedia url

        return dataset
        

#truthful_qa
class truthful_qa(Processor):
    """
    DatasetDict({
    validation: Dataset({
        features: ['type', 'category', 'question', 'best_answer', 'correct_answers', 'incorrect_answers', 'source'],
        num_rows: 817
    })
})

    """
    wiki_api = "https://en.wikipedia.org/w/api.php?action=query&format=json&titles={}"
    def __init__(self, *args, **kwargs):
        dataset_name = 'truthful_qa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    @staticmethod
    def fetch_wiki_id(inp):
        wiki_link, wiki_title = inp
        response = requests.get(truthful_qa.wiki_api.format(wiki_title))
        wiki_object_has_no_pages = 0
        if response.status_code == 200:
            data = response.json()
            # hack because of weird dict structure returned json of the wiki_api
            pages = data.get("query", {}).get("pages", {})
            if pages:
                wiki_id = next(iter(pages.keys()))
                return wiki_id
            else:
                wiki_object_has_no_pages += 1
                return ""
        else:
            count_not_found += 1
            print(f"wiki page {wiki_link} could not be fetched from wiki!")
            return ""


    def process(self):

        hf_name = 'truthful_qa' 
        dataset = datasets.load_dataset(hf_name, "generation",num_proc=self.num_proc)[self.split]
        

        # use "sample_id" ?
        dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)

        # dataset = dataset.rename_column("best_answer", "label")
        dataset = dataset.map(lambda example: {'label': [example['best_answer']]})
        dataset = dataset.rename_column("question", "content")

        def get_wiki_id(example):
            wiki_link = example['source']
            wiki_title = wiki_link.split("/")[-1]
            id= truthful_qa.fetch_wiki_id((wiki_link, wiki_title))
            return truthful_qa.fetch_wiki_id((wiki_link, wiki_title))

        #dataset = dataset.map(lambda example: {'ranking_label': get_wiki_id(example)},num_proc=5)

        dataset = dataset.remove_columns([ 'best_answer', 'type','category', 'correct_answers','incorrect_answers','source'])

        # ranking_label: wikipedia url

        return dataset
        
class POPQA(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'popqa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'akariasai/PopQA' 
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("question", "content")
        dataset = dataset.map(lambda example: {'label': eval(example['possible_answers'])})
        dataset = dataset.remove_columns(["possible_answers","id","subj", "prop","obj","subj_id","prop_id",'obj_id','s_aliases','o_aliases','s_uri','o_uri','s_wiki_title','o_wiki_title','s_pop','o_pop'])
        #generating the id, train_0 ... validation_0 validation_1
        cid= [self.split+str(i) for i in range(len(dataset))]
        dataset = dataset.add_column("id", cid)

        return dataset

class wikimultihopqa(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = '2wikimultihopqa'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35' 
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("question", "content")
        #dataset = dataset.rename_column("answer", "label")
        dataset = dataset.map(lambda example: {'label': [example['answer']]})
        dataset = dataset.remove_columns(["answer","evidences", "supporting_facts","context"])
        #generating the id, train_0 ... validation_0 validation_1
        cid= [self.split+str(i) for i in range(len(dataset))]
        dataset = dataset.add_column("id", cid)
        
        return dataset
   

class MMLU(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'mmlu'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'cais/mmlu' 
        hf_query_or_doc_name= "all"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]

        def make_question(x):
            choices_str = ' '.join([f"{i+1}. {c}" for i, c in enumerate(x['choices'])])
            return {'content': f"{x['question']} {choices_str}"}
        # use "sample_id" ?
        dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)
                
        dataset = dataset.map(make_question, num_proc=self.num_proc, batched=False)
        dataset = dataset.rename_column("answer", "label")
        dataset = dataset.remove_columns(['subject', 'choices', 'question'])

        # example after processing
        
        # {'question': "Box a nongovernmental not-for-profit organization had the following transactions 
        # during the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment
        # $10000 Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be 
        # reported as net cash provided by financing activities in Box's statement of cash flows?", 'label': 3,
        # 'content': "Box a nongovernmental not-for-profit organization had the following transactions during
        # the year: Proceeds from sale of investments $80000 Purchase of property plant and equipment $10000 
        # Proceeds from long-term debt $100000 Loss on sale of investment $5000 What amount should be reported 
        # as net cash provided by financing activities in Box's 
        # statement of cash flows? 1. $70,000 2. $75,000 3. $80,000 4. 100000"}

        return dataset

class NQOpen(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'nq_open'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        

    def process(self):

        hf_name = 'nq_open' 
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]

        

        dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)

        dataset = dataset.rename_column("answer", "label")
        dataset = dataset.rename_column("question", "content")
        # replace \xa0 chars with space
        # Define a function to clean the labels
        def clean_labels(labels):
            return [label.replace('\xa0', ' ') for label in labels]

        # Apply the cleaning function to the 'label' column
        dataset = dataset.map(lambda example: {'label': clean_labels(example['label'])})

        return dataset

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
    
class MsMarcoQueries(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'ms-marco-dev-queries'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        queries_d = json.load(open("/gfs-ssd/user/tformal/neural_search/MSMARCO/dev_queries_collection/dev_queries.json"))  # super hard-coded path, see how to do properly
        ids, queries = zip(*queries_d.items())
        dataset = datasets.Dataset.from_dict({"id":ids, "content": queries})  # no need for split?
        return dataset

class MKQA(Processor):

    def __init__(self, lang, *args, **kwargs):
        dataset_name = f'mkqa_{lang}'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang
        
    def process(self):
        mkqa = datasets.load_dataset('mkqa')
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
        os.system("wget https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl")
        dataset = datasets.load_dataset("json", data_files="xor_dev_full_v1_1.jsonl")["train"] # the file should be already .dev, and train is just default hf label
        dataset = dataset.filter(lambda example: example['lang']==self.lang)
        # discarding empty answers 
        dataset = dataset.map(lambda example: {'label': [el for el in example['answers'] if len(el) > 0]})
        dataset = dataset.rename_column("question", "content")
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

# ---------------------------------------- #
# Document processors
# ---------------------------------------- #
class ReproduceWikiCorpora63(Processor):

    def __init__(self, data_path, label="", *args, **kwargs):
        self.dataset_name = 'reproduce-wiki-corpora-63'
        if label != "":
            self.dataset_name = self.dataset_name + label
        self.path = data_path
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        dataset = datasets.load_dataset("csv", data_files=[self.path], delimiter="\t", column_names=["id", "text", "title"])[self.split]
                
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset
    
class ODQAWikiCorpora100WTamber(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'odqa-wiki-corpora-100w-tamber'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        

    def process(self):
        hf_name = 'castorini/odqa-wiki-corpora'
        hf_query_or_doc_name= "wiki-text-100w-tamber"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        def map_fn(example):
            example['content'] = f"{example['title']} {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.rename_column("docid", "id")
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset
    
class KILT100w(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'kilt-100w'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        hf_name = 'kilt_wikipedia'
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]

        def map_100w(sample, num_words=100):
            wiki_id = sample['wikipedia_id']
            title = sample['wikipedia_title']
            passages = [x.strip() for x in sample["text"]["paragraph"] if "BULLET::::" not in x]
            doc = " ".join(passages)
            doc = doc.replace('Section::::', 'Section:')
            words = doc.split()
            paragraphs = [title + '. ' + " ".join(words[i:i + num_words]) for i in range(0, len(words), num_words)]
            wiki_ids = [wiki_id] * len(paragraphs)
            return {'paragraphs': paragraphs, "wiki_ids": wiki_ids}
        
        def map_predefined_paragraphs(sample):
            wiki_id = sample['wikipedia_id']
            title = sample['wikipedia_title']
            paragraphs = [title + '. ' + el.replace('Section::::', 'Section:') for el in sample["text"]["paragraph"]]
            ids = [f'{wiki_id}_{i+1}' for i in range(len(paragraphs))]
            wiki_ids = [wiki_id] * len(paragraphs)
            return {'paragraphs': paragraphs, "id": ids, "wiki_ids": wiki_ids}
        
        map_fn = map_predefined_paragraphs if self.oracle_provenance else map_100w
        kilt_dataset = dataset.map(map_fn, num_proc=self.num_proc)
        paragraphs = [el for sublist in kilt_dataset['paragraphs'] for el in sublist]

        if self.oracle_provenance:
            ids = [el for sublist in kilt_dataset['id'] for el in sublist]
            dataset = datasets.Dataset.from_dict({'content': paragraphs, 'id': ids})
        else:
            wiki_ids = [el for sublist in kilt_dataset['wiki_ids'] for el in sublist]
            dataset = datasets.Dataset.from_dict({'content': paragraphs, 'wikipedia_id': wiki_ids})
            dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)

        del kilt_dataset
        return dataset

class Wiki_monolingual_100w(Processor):

    def __init__(self, lang, *args, **kwargs):
        dataset_name = 'wiki-100w-' + lang
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.lang = lang

    def process(self):
        hf_name = 'wikimedia/wikipedia'
        subset = "20231101." + self.lang
        dataset = datasets.load_dataset(hf_name, subset, num_proc=self.num_proc)[self.split]

        def map_100w(sample, num_words=100):
            wiki_id = sample['id']
            title = sample['title']
            doc = sample["text"]
            if self.lang not in ["zh", "ja", "th"]:
                words = doc.split()
            else:
                words = list(doc)
            paragraphs = [title + '. ' + " ".join(words[i:i + num_words]) for i in range(0, len(words), num_words)]
            wiki_ids = [wiki_id] * len(paragraphs)
            return {'paragraphs': paragraphs, "wiki_ids": wiki_ids}
        
        kilt_dataset = dataset.map(map_100w, num_proc=self.num_proc)
        paragraphs = [el for sublist in kilt_dataset['paragraphs'] for el in sublist]
        wiki_ids = [el for sublist in kilt_dataset['wiki_ids'] for el in sublist]
        dataset = datasets.Dataset.from_dict({'content': paragraphs, 'wikipedia_id': wiki_ids})
        dataset = dataset.map(lambda example, idx: {'id': str(idx), **example}, with_indices=True)

        del kilt_dataset
        return dataset

class ODQAWikiCorpora100WKarpukhin(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'odqa-wiki-corpora-100w-karpukhin'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        


    def process(self):
        hf_name = 'castorini/odqa-wiki-corpora'
        hf_query_or_doc_name= "wiki-text-100w-karpukhin"
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.rename_column("docid", "id")
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset
    
class ODQAWikiCorpora63tamber(Processor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_name = 'odqa-wiki-corpora-63-tamber'
    def process(self):
        hf_name = 'castorini/odqa-wiki-corpora'
        hf_query_or_doc_name= "wiki-text-6-3-tamber"
        
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.rename_column("docid", "id")
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset


class ODQAWikiCorpora63tamberALL(Processor):

    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.dataset_name = 'odqa-wiki-corpora-all-63-tamber'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)

    def process(self):
        hf_name = 'castorini/odqa-wiki-corpora'
        hf_query_or_doc_name= "wiki-all-6-3-tamber"
        
        dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.rename_column("docid", "id")
        #dataset = dataset.map(lambda example: {'wikipedia_id': example['id'].split("#")[0] })
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset


class PubMed2023(Processor):

    def __init__(self, data_path, *args, **kwargs):
        self.dataset_name = 'PubMed-2023'
        self.path = data_path
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        dataset = datasets.load_dataset("csv", data_files=[self.path], delimiter="\t", column_names=["id", "title", "text"])[self.split]
                
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.remove_columns(['title', 'text'])
        return dataset
    
    



 
class MsMarcoCollection(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'ms-marco'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        

    def process(self):
        # load from the ir-dataset HF repo
        hf_name = "irds/msmarco-passage"
        dataset = datasets.load_dataset(hf_name, 'docs', num_proc=self.num_proc)  # no need for split?
        dataset = dataset.rename_column("doc_id", "id")
        dataset = dataset.rename_column("text", "content")
        return dataset

# applies processing to dataset names
# processes query and doc with different processors


class UT1Queries(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'ut1queries'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        data_d={
            'id':[],
            'content':[],
            'label':[],
        }
        f=open('tests/utdata/ut1_queries.tsv')
        for l in f:
            tok=l.split('\t')
            qid=tok[0].strip()
            qt=tok[1].strip()
            label=tok[2].strip()
            data_d['id'].append(qid)
            data_d['content'].append(qt)
            data_d['label'].append(label)
        dataset=datasets.Dataset.from_dict(data_d)
        return dataset

class UT1Docs(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'ut1docs'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        data_d={
            'id':[],
            'content':[],
        }
        f=open('tests/utdata/ut1_docs.tsv')
        for l in f:
            tok=l.split('\t')
            did=tok[0].strip()
            dt=tok[1].strip()
            data_d['id'].append(did)
            data_d['content'].append(dt)
        dataset=datasets.Dataset.from_dict(data_d)
        return dataset


class MergedDocDataset(Processor):
    def __init__(self, out_dataset_name, in_dataset_names, in_dataset_splits, *args, **kwargs):
        dataset_name = out_dataset_name
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        self.in_dataset_names = in_dataset_names
        self.in_dataset_splits = in_dataset_splits
        assert len(in_dataset_names) == len(in_dataset_splits)

    def process(self):
        raise NotImplementedError("Datasets for merging should be preprocessed independently before runing experiments with the merged dataset.")

    def get_dataset(self):
        def prepend_label(example, label):
            example['id'] = f"{label}_{example['id']}"
            return example
        print(f"Processing dataset {self.dataset_name} in {self.split} split ")
        debug_str = '_debug' if self.debug else ''
        assert self.dataset_name != None # dataset name needs to be set in processor class
        oracle_provenance_str = '_oracle_provenance' if self.oracle_provenance else ''
        out_folder = os.path.join(f'{self.out_folder}', f'{self.dataset_name}_{self.split}{oracle_provenance_str}')

        loaded_datasets = []
        for dataset_name, split in zip(self.in_dataset_names, self.in_dataset_splits):
            in_folder = os.path.join(f'{self.out_folder}', f'{dataset_name}_{split}{oracle_provenance_str}')
            if not os.path.exists(in_folder): raise ValueError(f"Dataset {in_folder} not found")
            dataset = datasets.load_from_disk(in_folder)
            dataset = dataset.map(partial(prepend_label, label=dataset_name), num_proc=self.num_proc)
            loaded_datasets.append(dataset)
            
        dataset = datasets.concatenate_datasets(loaded_datasets)
        
        id2index = self.get_index_to_id(dataset) 
        dataset.id2index = id2index
        if self.debug:
            dataset = dataset.select(range(15))
        if self.shuffle_labels:
            dataset = self.shuffled_labels_as_content(dataset)
        dataset.name = self.dataset_name + debug_str + oracle_provenance_str
        return dataset


class ProcessDatasets:
                
    @staticmethod
    def process(datasets, out_folder='datasets', num_proc=1, overwrite=False, debug=False, oracle_provenance=False, shuffle_labels=False):
        def sanity_checks(dataset):
            for example in tqdm(dataset, 'Checking dataset..'):
                for field_name, field_value in example.items():
                    if field_value is None:
                        raise ValueError(f"Found None value in '{field_name}' field.")
                    elif isinstance(field_value, list) and None in field_value:
                        raise ValueError(f"Found None in list in '{field_name}' field.")
                    elif isinstance(field_value, str) and len(field_value.strip()) == 0:
                        raise ValueError(f"Found empty value in '{field_name}' field.")
                    elif isinstance(field_value, list) and len(field_value) == 0:
                        raise ValueError(f"Found empty list in '{field_name}' field.")
                

        processed_datasets = defaultdict(dict)
        for split in datasets:
            for query_or_doc in datasets[split]:
                if datasets[split][query_or_doc] != None:
                    processor_init_args = datasets[split][query_or_doc]['init_args']
                    processor = instantiate(
                        processor_init_args, 
                        out_folder=out_folder, 
                        num_proc=num_proc, 
                        overwrite=overwrite, 
                        debug= debug if query_or_doc == 'query' else False, 
                        oracle_provenance= oracle_provenance if query_or_doc == 'doc' else False, 
                        shuffle_labels= shuffle_labels if query_or_doc == 'query' else False
                        )
                    dataset = processor.get_dataset()
                    if query_or_doc == 'query':
                        sanity_checks(dataset)
                    processed_datasets[split][query_or_doc] = dataset
                else:
                    processed_datasets[split][query_or_doc] = None
        return processed_datasets

    
