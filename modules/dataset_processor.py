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
                shuffle_labels,
                idx_min: int = None,
                idx_max: int = None
                ):
        self.dataset_name = dataset_name
        self.split = split
        self.num_proc = num_proc
        self.out_folder = out_folder
        self.overwrite = overwrite
        self.debug = debug
        self.oracle_provenance = oracle_provenance
        self.shuffle_labels = shuffle_labels
        self.use_cache = True # if set to False, then no cache file is read or written for the dataset
        
        # Ids used to work only with a subset of the dataset: we'll select range(idx_min, idx_max)
        self.idx_min = idx_min
        self.idx_max = idx_max

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
                dataset = dataset.select(range(15))
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
        
        # If ids are provided, we select the corresponding range:
        # We then filter the 'id2index' to keep only ids/index present in the dataset
        idx_min = self.idx_min or 0
        idx_max = self.idx_max or len(dataset)
        if idx_max == -1:
            idx_max = len(dataset)
        assert 0 <= idx_min < len(dataset), f"{idx_min} but {len(dataset)}"
        assert 0 < idx_max <= len(dataset), f"{idx_max} but {len(dataset)}"
        if idx_min != 0 or idx_max != len(dataset):
            print(f"Selecting {idx_min} to {idx_max} in dataset {self.dataset_name} of length {len(dataset)}")
            dataset = dataset.select(range(idx_min, idx_max))
            dataset.id2index = self.get_index_to_id(dataset) # we recompute it
        else:
            dataset.id2index = id2index
            
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
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
                
        def map_fn(example):
            example['content'] = f"{example['MedlineCitation']['Article']['ArticleTitle']}: {example['MedlineCitation']['Article']['Abstract']['AbstractText']}"
            example['id'] = str(example['MedlineCitation']['PMID'])
            return example
        
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.remove_columns(['MedlineCitation', 'PubmedData'])
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
                        oracle_provenance=oracle_provenance if query_or_doc == 'doc' else False, 
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


class KILTMULTIQA(Processor):
    def __init__(self, response_files: list = None, *args, **kwargs):
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
        
        def replace_label(example):
            example['label'] = [new_data[example['id']]]
            return example
                    
        print('Replacing labels in dataset with read responses...')
        dataset = dataset.map(replace_label)
        
        return dataset

    
