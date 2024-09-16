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
import pandas as pd

from urllib.parse import unquote
from bs4 import BeautifulSoup

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
        # ranking_label: list of wikipedia_ids per answer, empty list if no provenances are present or answer is empty
        # No ranking labels
        #dataset = dataset.map(lambda example: {'ranking_label': [[provenance['wikipedia_id'] for provenance in el['provenance']] if len(el['answer']) > 0 and len(el['provenance']) > 0 else [] for el in example['output']]})

    
        return dataset

class BIOASQ11B_Ragged(Processor):
    """BIOASQ benchmark, dataset from Ragged paper"""

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'BIOASQ11B_Ragged'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name = "jenhsia/ragged"
        dataset = datasets.load_dataset(hf_name, 'bioasq', num_proc=self.num_proc)[self.split]
        # ['id', 'input', 'output', 'question_type']
        dataset = dataset.map(lambda example: {'label': [dictt["answer"] for dictt in example["output"] if dictt["answer"] is not None]})
        dataset = dataset.map(lambda example: {'label': [" ".join(example['label'])] if example['question_type'] == 'list' else example['label']})  # concatenate list question types to compute recall on all labels (see metrics computation)
        dataset = dataset.rename_column("input", "content")
        dataset = dataset.remove_columns(['question_type', 'output'])
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
        #dataset = dataset.rename_column("answer", "label")
        dataset = dataset.map(lambda example: {"label": [str(example["answer"])]})
        dataset = dataset.remove_columns(['subject', 'choices', 'question','answer'])

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

#FIME does not seem to be used anymore?
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

    
class MsMarcoQueries(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'ms-marco-dev-queries'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        queries_d = json.load(open("/gfs-ssd/user/tformal/neural_search/MSMARCO/dev_queries_collection/dev_queries.json"))  # super hard-coded path, see how to do properly
        ids, queries = zip(*queries_d.items())
        dataset = datasets.Dataset.from_dict({"id":ids, "content": queries})  # no need for split?
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

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'PubMed-2023'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name ="ncbi/pubmed"
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc, trust_remote_code=True)[self.split]
                
        def map_fn(example):
            example['content'] = f"{example['MedlineCitation']['Article']['ArticleTitle']}: {example['MedlineCitation']['Article']['Abstract']['AbstractText']}"
            example['id'] = str(example['MedlineCitation']['PMID'])
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.remove_columns(['MedlineCitation', 'PubmedData'])
        return dataset


class PubMed2023_Ragged(Processor):
    """PubMed abstracts, dataset from Ragged paper"""

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'PubMed-2023_Ragged'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name = "jenhsia/ragged"
        dataset = datasets.load_dataset(hf_name, 'pubmed', num_proc=self.num_proc)[self.split]

        concatenated_data = {}
        for row in tqdm(dataset):
            real_id, field_type = row['id'].split('_')

            if real_id not in concatenated_data:
                concatenated_data[real_id] = {"title": "", "content": ""}
            
            if field_type == '0':
                concatenated_data[real_id]["title"] = row['contents']
            elif field_type == '1':
                concatenated_data[real_id]["content"] = row['contents']
                
        concatenated_rows = []
        for real_id, fields in tqdm(concatenated_data.items()):
            title = fields["title"]
            content = fields["content"]
            concatenated_content = f"{title}: {content}" if content else title
            concatenated_rows.append({"id": real_id, "content": concatenated_content})

        dataset = datasets.Dataset.from_list(concatenated_rows)

        return dataset


class Wikipedia2023_section(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'wikipedia-2023-section'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name = 'rasdani/cohere-wikipedia-2023-11-en'  #from Cohere/wikipedia-2023-11-embed-multilingual-v3 w/o embeddings
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
                
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.remove_columns(['text', 'title'])
        dataset = dataset.rename_column("_id", "id")
        return dataset    
    


class Wikipedia2023_full(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'wikipedia-2023-full'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name = 'wikimedia/wikipedia'
        dataset = datasets.load_dataset(hf_name, '20231101.en',num_proc=self.num_proc)[self.split]
                
        def map_fn(example):
            example['content'] = f"{example['title']}: {example['text']}"
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.remove_columns(['id','text', 'title'])
        dataset = dataset.rename_column("url", "id")
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
        script_path = os.path.abspath(__file__)
        tests_directory = os.path.join(os.path.dirname(os.path.dirname(script_path)), 'tests')
        with open(os.path.join(tests_directory,'utdata/ut1_queries.tsv'), 'r') as f:
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
        # We get the adress of the tests tsv file, relative to this script path:
        script_path = os.path.abspath(__file__)
        tests_directory = os.path.join(os.path.dirname(os.path.dirname(script_path)), 'tests')
        
        with open(os.path.join(tests_directory, 'utdata/ut1_docs.tsv'), 'r') as f:
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


class APIBench_gorilla_HF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_HF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/huggingface_eval.json'

        def download_file(url):
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            data = []
            for line in response.text.strip().split('\n'):
                data.append(json.loads(line))
            return data

        def index(data):
            """
            data: list of python dictionaries
            """
            for i,_ in enumerate(data):
                data[i]['id'] = i
                if 'performance' in data[i]:
                    del data[i]['performance']
            return data

        api_data = download_file(apibench_file)
        
        print('Processing...')
        indexed_api_bench = index(api_data)
        tmp_df = pd.DataFrame(data=indexed_api_bench)

        def get_instruction(x):
            tmp = x.split('###Instruction:')
            if len(tmp) == 2:
                tmp = tmp[1].split('###Output:')
                return tmp[0].strip().replace('\n', '\\')
            else:
                tmp = x.split('### Instruction:')
                if len(tmp) == 2:
                    tmp = tmp[1].split('###Output:')
                    if len(tmp) >= 2:
                        return tmp[0].strip().replace('\n', '\\')
                    else:
                        tmp = tmp[0].split('### Output:')
                        if len(tmp) >= 2:
                            return tmp[0].strip().replace('\n', '\\')
                else:
                    return None
            
        tmp_df['content'] = tmp_df['code'].apply(get_instruction)
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        tmp_df['label'] = tmp_df['api_call']
        tmp_df = tmp_df.drop(['code', 'provider', 'api_data'], axis=1).dropna()
        api_bench_dataset = datasets.Dataset.from_pandas(tmp_df)
        if "__index_level_0__" in api_bench_dataset.column_names:
            api_bench_dataset = api_bench_dataset.remove_columns(["api_call", "__index_level_0__"]).cast_column('id', datasets.Value('string')).map(listify_label)
        else:
            api_bench_dataset = api_bench_dataset.remove_columns(["api_call"]).cast_column('id', datasets.Value('string')).map(listify_label)
        print('Done.')
        return api_bench_dataset
    
class APIBench_gorilla_TF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_TF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/tensorflow_eval.json'

        def download_file(url):
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            data = []
            for line in response.text.strip().split('\n'):
                data.append(json.loads(line))
            return data

        def index(data):
            """
            data: list of python dictionaries
            """
            for i,_ in enumerate(data):
                data[i]['id'] = i
                if 'performance' in data[i]:
                    del data[i]['performance']
            return data

        api_data = download_file(apibench_file)
        
        print('Processing...')
        indexed_api_bench = index(api_data)
        tmp_df = pd.DataFrame(data=indexed_api_bench)

        def get_instruction(x):
            tmp = x.split('###Instruction:')
            if len(tmp) == 2:
                tmp = tmp[1].split('###Output:')
                return tmp[0].strip().replace('\n', '\\')
            else:
                tmp = x.split('### Instruction:')
                if len(tmp) == 2:
                    tmp = tmp[1].split('###Output:')
                    if len(tmp) >= 2:
                        return tmp[0].strip().replace('\n', '\\')
                    else:
                        tmp = tmp[0].split('### Output:')
                        if len(tmp) >= 2:
                            return tmp[0].strip().replace('\n', '\\')
                else:
                    return None
            
        tmp_df['content'] = tmp_df['code'].apply(get_instruction)
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        tmp_df['label'] = tmp_df['api_call']
        tmp_df = tmp_df.drop(['code', 'provider', 'api_data'], axis=1).dropna()
        api_bench_dataset = datasets.Dataset.from_pandas(tmp_df)
        if "__index_level_0__" in api_bench_dataset.column_names:
            api_bench_dataset = api_bench_dataset.remove_columns(["api_call", "__index_level_0__"]).cast_column('id', datasets.Value('string')).map(listify_label)
        else:
            api_bench_dataset = api_bench_dataset.remove_columns(["api_call"]).cast_column('id', datasets.Value('string')).map(listify_label)
        print('Done.')
        return api_bench_dataset
    
class APIBench_gorilla_TH(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_TH'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/torchhub_eval.json'

        def download_file(url):
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            data = []
            for line in response.text.strip().split('\n'):
                data.append(json.loads(line))
            return data

        def index(data):
            """
            data: list of python dictionaries
            """
            for i,_ in enumerate(data):
                data[i]['id'] = i
                if 'performance' in data[i]:
                    del data[i]['performance']
            return data

        api_data = download_file(apibench_file)
        
        print('Processing...')
        indexed_api_bench = index(api_data)
        tmp_df = pd.DataFrame(data=indexed_api_bench)

        def get_instruction(x):
            tmp = x.split('###Instruction:')
            if len(tmp) == 2:
                tmp = tmp[1].split('###Output:')
                return tmp[0].strip().replace('\n', '\\')
            else:
                tmp = x.split('### Instruction:')
                if len(tmp) == 2:
                    tmp = tmp[1].split('###Output:')
                    if len(tmp) >= 2:
                        return tmp[0].strip().replace('\n', '\\')
                    else:
                        tmp = tmp[0].split('### Output:')
                        if len(tmp) >= 2:
                            return tmp[0].strip().replace('\n', '\\')
                else:
                    return None
            
        tmp_df['content'] = tmp_df['code'].apply(get_instruction)
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        tmp_df['label'] = tmp_df['api_call']
        tmp_df = tmp_df.drop(['code', 'provider', 'api_data'], axis=1).dropna()
        api_bench_dataset = datasets.Dataset.from_pandas(tmp_df)
        if "__index_level_0__" in api_bench_dataset.column_names:
            api_bench_dataset = api_bench_dataset.remove_columns(["api_call", "__index_level_0__"]).cast_column('id', datasets.Value('string')).map(listify_label)
        else:
            api_bench_dataset = api_bench_dataset.remove_columns(["api_call"]).cast_column('id', datasets.Value('string')).map(listify_label)
        print('Done.')
        return api_bench_dataset
    

class API_gorilla_HF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_HF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """

        api_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/huggingface_api.jsonl'

        def download_file(url):
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            data = []
            for line in response.text.strip().split('\n'):
                data.append(json.loads(line))
            return data
        
        def index(data):
            """
            data: list of python dictionaries
            """
            for i,_ in enumerate(data):
                data[i]['id'] = i
                if 'performance' in data[i]:
                    del data[i]['performance']
            return data
        
        def convert_to_string(x):
            if isinstance(x, list):  # Check if the element is a list
                if len(x) > 0:
                    if isinstance(x[0], dict):
                        return str(x)
                return ','.join(x)   # Join the list elements with commas
            return str(x)                 # If it's not a list, return the original string

        api_data = download_file(api_file)
        print('Processing...')
        indexed_api = index(api_data) # create indices
        tmp_df = pd.DataFrame(data=indexed_api)
        tmp_df['api_arguments'] = tmp_df['api_arguments'].apply(convert_to_string)
        tmp_df['python_environment_requirements'] = tmp_df['python_environment_requirements'].apply(convert_to_string)
        tmp_df['example_code'] = tmp_df['example_code'].apply(convert_to_string)
        tmp_df['functionality'] = tmp_df['functionality'].apply(convert_to_string)
        api_dataset = datasets.Dataset.from_pandas(tmp_df).map(lambda row: {'content':'\n'.join([f"{key}: {value};" for key,value in row.items() if key != 'id'])})
        api_dataset = api_dataset.remove_columns([column for column in api_dataset.column_names if column not in ['id', 'content']]).cast_column('id', datasets.Value('string'))
        print('Done.')
        return api_dataset
    

class API_gorilla_TF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_TF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """

        api_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/tensorflowhub_api.jsonl'

        def download_file(url):
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            data = []
            for line in response.text.strip().split('\n'):
                data.append(json.loads(line))
            return data
        
        def index(data):
            """
            data: list of python dictionaries
            """
            for i,_ in enumerate(data):
                data[i]['id'] = i
                if 'performance' in data[i]:
                    del data[i]['performance']
            return data
        
        def convert_to_string(x):
            if isinstance(x, list):  # Check if the element is a list
                if len(x) > 0:
                    if isinstance(x[0], dict):
                        return str(x)
                return ','.join(x)   # Join the list elements with commas
            return str(x)                 # If it's not a list, return the original string

        api_data = download_file(api_file)
        print('Processing...')
        indexed_api = index(api_data) # create indices
        tmp_df = pd.DataFrame(data=indexed_api)
        tmp_df['api_arguments'] = tmp_df['api_arguments'].apply(convert_to_string)
        tmp_df['python_environment_requirements'] = tmp_df['python_environment_requirements'].apply(convert_to_string)
        tmp_df['example_code'] = tmp_df['example_code'].apply(convert_to_string)
        tmp_df['functionality'] = tmp_df['functionality'].apply(convert_to_string)
        api_dataset = datasets.Dataset.from_pandas(tmp_df).map(lambda row: {'content':'\n'.join([f"{key}: {value};" for key,value in row.items() if key != 'id'])})
        api_dataset = api_dataset.remove_columns([column for column in api_dataset.column_names if column not in ['id', 'content']]).cast_column('id', datasets.Value('string'))
        print('Done.')
        return api_dataset
    
class API_gorilla_TH(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_TH'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """

        api_file = f'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/torchhub_api.jsonl'

        def download_file(url):
            print(f"Downloading {url}...")
            response = requests.get(url)
            response.raise_for_status()  # Check if the request was successful
            data = []
            for line in response.text.strip().split('\n'):
                data.append(json.loads(line))
            return data
        
        def index(data):
            """
            data: list of python dictionaries
            """
            for i,_ in enumerate(data):
                data[i]['id'] = i
                if 'performance' in data[i]:
                    del data[i]['performance']
            return data
        
        def convert_to_string(x):
            if isinstance(x, list):  # Check if the element is a list
                if len(x) > 0:
                    if isinstance(x[0], dict):
                        return str(x)
                return ','.join(x)   # Join the list elements with commas
            return str(x)                 # If it's not a list, return the original string

        api_data = download_file(api_file)
        print('Processing...')
        indexed_api = index(api_data) # create indices
        tmp_df = pd.DataFrame(data=indexed_api)
        tmp_df['api_arguments'] = tmp_df['api_arguments'].apply(convert_to_string)
        tmp_df['python_environment_requirements'] = tmp_df['python_environment_requirements'].apply(convert_to_string)
        tmp_df['example_code'] = tmp_df['example_code'].apply(convert_to_string)
        tmp_df['functionality'] = tmp_df['functionality'].apply(convert_to_string)
        api_dataset = datasets.Dataset.from_pandas(tmp_df).map(lambda row: {'content':'\n'.join([f"{key}: {value};" for key,value in row.items() if key != 'id'])})
        api_dataset = api_dataset.remove_columns([column for column in api_dataset.column_names if column not in ['id', 'content']]).cast_column('id', datasets.Value('string'))
        print('Done.')
        return api_dataset
    

class CodeRAGBench_HumanEval(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = "CodeRAGBench_HumanEval"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        # load from the CodeRAGBench HF repo
        hf_name = "code-rag-bench/humaneval"
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("task_id", "id").rename_column("prompt", "content").rename_column("canonical_solution", "label")
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        dataset = dataset.map(listify_label)
        return dataset


class CodeRAGBench_MBPP(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = "CodeRAGBench_MBPP"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        # load from the CodeRAGBench HF repo
        hf_name = "code-rag-bench/mbpp"
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("task_id", "id").rename_column("text", "content").rename_column("code", "label")
        dataset = dataset.remove_columns([column for column in dataset.column_names if column not in ['id', 'content', 'label']])
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        dataset = dataset.map(listify_label)
        return dataset

class CodeRAGBench_database(Processor):
    """All docs from CodeRAGBench paper -- programming solutions, github repos, stack-overflow, tutorials, library docs"""

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_database'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):

        def cat_title_content(x, content_colname="content", title_colname="title"):
            if title_colname is None:
                x["content"] = f"{x[content_colname]}"
            else:
                x["content"] = f"{x[title_colname]}: {x[content_colname]}"
            return x
        
        dataset1 = datasets.load_dataset("code-rag-bench/programming-solutions", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset2 = datasets.load_dataset("code-rag-bench/online-tutorials", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset3 = datasets.load_dataset("code-rag-bench/library-documentation", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"doc_content", "title_colname":"doc_id"}).select_columns(['content'])
        dataset4 = datasets.load_dataset("code-rag-bench/stackoverflow-posts", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset5 = datasets.load_dataset("code-rag-bench/github-repos-python", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset6 = datasets.load_dataset("code-rag-bench/github-repos", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])

        # concat and create ids
        dataset = datasets.concatenate_datasets([dataset1, dataset2, dataset3, dataset4, dataset5, dataset6]).map(lambda _, idx: {"id": str(idx)}, with_indices=True)

        return dataset

class CodeRAGBench_programming_solutions(Processor):
    """Contains oracle docs for HumanEval and MBPP"""
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_programming_solutions'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        def cat_title_content(x, content_colname="content", title_colname="title"):
            if title_colname is None:
                x["content"] = f"{x[content_colname]}"
            else:
                x["content"] = f"{x[title_colname]}: {x[content_colname]}"
            return x
        dataset = datasets.load_dataset("code-rag-bench/programming-solutions", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_online_tutorials(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_online_tutorials'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        def cat_title_content(x, content_colname="content", title_colname="title"):
            if title_colname is None:
                x["content"] = f"{x[content_colname]}"
            else:
                x["content"] = f"{x[title_colname]}: {x[content_colname]}"
            return x
        dataset = datasets.load_dataset("code-rag-bench/online-tutorials", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_library_documentation(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_library_documentation'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        def cat_title_content(x, content_colname="content", title_colname="title"):
            if title_colname is None:
                x["content"] = f"{x[content_colname]}"
            else:
                x["content"] = f"{x[title_colname]}: {x[content_colname]}"
            return x
        dataset = datasets.load_dataset("code-rag-bench/library-documentation", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"doc_content", "title_colname":"doc_id"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_stackoverflow(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_stackoverflow'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        def cat_title_content(x, content_colname="content", title_colname="title"):
            if title_colname is None:
                x["content"] = f"{x[content_colname]}"
            else:
                x["content"] = f"{x[title_colname]}: {x[content_colname]}"
            return x
        dataset = datasets.load_dataset("code-rag-bench/stackoverflow-posts", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset

class CodeRAGBench_gitrepospython(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_gitrepospython'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        def cat_title_content(x, content_colname="content", title_colname="title"):
            if title_colname is None:
                x["content"] = f"{x[content_colname]}"
            else:
                x["content"] = f"{x[title_colname]}: {x[content_colname]}"
            return x
        dataset = datasets.load_dataset("code-rag-bench/github-repos-python", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_gitrepos(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_gitrepos'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        def cat_title_content(x, content_colname="content", title_colname="title"):
            if title_colname is None:
                x["content"] = f"{x[content_colname]}"
            else:
                x["content"] = f"{x[title_colname]}: {x[content_colname]}"
            return x
        dataset = datasets.load_dataset("code-rag-bench/github-repos", num_proc=self.num_proc)[self.split].map(cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset

class SyllabusQA(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = "SyllabusQA"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        url = "https://raw.githubusercontent.com/umass-ml4ed/SyllabusQA/main/data/dataset_split/test.csv"
        df = pd.read_csv(url)
        def merge_syllabusname_question(row):
            row['content'] = row['syllabus_name'] + ": " + row['question']
            return row
        dataset = datasets.Dataset.from_pandas(df).map(merge_syllabusname_question).rename_column("answer", "label").remove_columns([
            'answer_span_1',
            'answer_span_2',
            'answer_span_3',
            'answer_span_4',
            'answer_span_5',
            'reasoning_step_1',
            'reasoning_step_2',
            'reasoning_step_3',
            'reasoning_step_4',
            'reasoning_step_5',
            ])
        def listify_label(row):
            row['label'] = [row['label']]
            return row
        dataset = dataset.map(listify_label)
        return dataset
    
class SyllabusQA_syllabi(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = "SyllabusQA_syllabi"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        Important notice: 
        Due to requests feature, sometimes this function fails with ```datasets.arrow_writer.SchemaInferenceError: Please pass `features` or at least one example when writing data```
        This happens when the request fails to retrieve all the txt files (and retrieves none of them) resulting in an empty Dataset
        Just run the function until it works... BUT make sure to remove the directory SyllabusQA_syllabi_train created in datasets/
        """

        url = "https://github.com/umass-ml4ed/SyllabusQA/tree/main/syllabi/syllabi_redacted/text"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        # parse the HTML to find all .txt file links
        base_url = "https://raw.githubusercontent.com/umass-ml4ed/SyllabusQA/main/syllabi/syllabi_redacted/text/"
        files = []

        # get file links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('.txt'):
                file_name = href.split('/')[-1]  # Get file name
                files.append(base_url + file_name)
        
        # duplicates
        files = list(set(files))
        print(len(files), 'files found.')

        # download each .txt
        syllabi = []
        for file_url in files:
            file_name = file_url.split('/')[-1]
            response = requests.get(file_url)
            content = response.content.decode('MacRoman')
            syllabi.append({'file_name': file_name, 'content': content})
            print(f"Downloaded {file_name}.")
        print("Done.")

        def chunk_text(text, title, max_size=1000, overlap=200):
            """
            Chunks the given text into parts with a maximum size and overlap, prepending the title to each chunk.
            
            Args:
            - text: The text to chunk.
            - title: The title of the syllabus to prepend to each chunk.
            - max_size: Maximum size of each chunk (default is 1000 characters, same as in https://arxiv.org/pdf/2403.14666).
            - overlap: Overlap between adjacent chunks (default is 200 characters, same as in https://arxiv.org/pdf/2403.14666).
            
            Returns:
            - A list of dictionaries with chunk 'id' and 'content' keys.
            """
            chunks = []
            start = 0
            chunk_id = 0
            while start < len(text):
                end = start + max_size
                chunk = text[start:end]
                chunk = title + ": " + chunk  # Prepend the title
                chunks.append({'id': f"{title}_{chunk_id}", 'content': chunk})
                start = end - overlap
                chunk_id += 1

            return chunks

        # chunk
        all_chunks = []
        for file in syllabi:
            title, text = unquote(file['file_name'].split('/')[-1].strip('.txt')), file['content']
            chunks = chunk_text(text, title)
            all_chunks.extend(chunks)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_chunks))
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
    
    @staticmethod
    def check_instantiate(datasets, out_folder='datasets', num_proc=1, overwrite=False, debug=False):
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
                        oracle_provenance=  False, 
                        shuffle_labels= False
                        )          
        return True


    
