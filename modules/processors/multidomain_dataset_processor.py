from ..dataset_processor import Processor
from modules.processors.utils import chunk_text, listify_label
import datasets
import json
import zipfile
import random

from tqdm import tqdm
import requests  
import pandas as pd
import os

from urllib.parse import unquote


class BIOASQ12B(Processor):
    """ 
    BIOASQ Benchmark from bioasq challenge source, year 2024 task B (12B)
    To get a larger training set we merge the official train and validation sets and fix the validation size to 1200 and train size to the rest (= 4189 rows)
    We then discard all 'summary' question types from the validation set yielding a final val set with 940 rows
    """

    def __init__(self, train_path, dev_path, *args, **kwargs):
        self.dataset_name = 'BIOASQ12B'
        self.train_path = train_path
        self.dev_path = dev_path
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)

    def process(self):
        seed = 42
        if self.split not in ["train", "dev"]:
            raise ValueError("split should be 'train' or 'dev'")
        all_data = []
        with zipfile.ZipFile(self.train_path, 'r') as z:
            with z.open('BioASQ-training12b/training12b_new.json') as json_file:
                all_data.extend(json.load(json_file)['questions'])
        with zipfile.ZipFile(self.dev_path, 'r') as z:
            for file_name in z.namelist():
                print(f"Loading file {file_name}")
                if file_name.endswith('.json'):
                    with z.open(file_name) as json_file:
                        all_data.extend(json.load(json_file)['questions'])
        random.seed(seed)
        random.shuffle(all_data)
        dev_data = all_data[:1200]
        train_data = all_data[1200:]
        if self.split == "train":
            data = train_data
        elif self.split == "dev":
            data = dev_data
        
        import itertools
        dataset = {"id": [], "content": [], "label": [], "type": []}
        for row in data:

            # parse labels
            if row['type'] == 'summary':
                if self.split == 'train':
                    if isinstance(row["ideal_answer"], list) and isinstance(row["ideal_answer"][0], str):
                        dataset['label'].append(row["ideal_answer"])
                    else:
                        raise ValueError(f"Unknown label structure for label {row['ideal_answer']}")
                elif self.split == 'dev': # discard summary questions for dev set
                    continue
            elif row['type'] == 'list':
                assert isinstance(row['exact_answer'], list) and isinstance(row['exact_answer'][0], list), f"unexpected parsing label for {row['id']}: {row['exact_answer']}"
                # put all combinations of needed answers x synonyms
                labels = [', '.join(combination) for combination in list(itertools.product(*row['exact_answer']))]
                if len(labels) > 1000:
                    print(f"WARNING: id={row['id']} is list-type label and has {len(labels)} combinations. Truncating to 10 synonyms max.")
                    labels = [', '.join(combination) for combination in list(itertools.product(*([e[:10] for e in row['exact_answer']])))]
                    if len(labels) > 1000:
                        print(f"    WARNING: After 10-truncation -> {len(labels)} labels. Truncating to 2 synonyms and 10 elements.")
                        labels = [', '.join(combination) for combination in list(itertools.product(*([e[:2] for e in row['exact_answer']][:10])))]
                        print(f"    WARNING: After final truncation -> {len(labels)} labels.")
                dataset["label"].append(labels)
            elif row['type'] == 'yesno':
                dataset['label'].append([row['exact_answer']])
            elif row['type'] == 'factoid':
                if isinstance(row['exact_answer'], list) and isinstance(row['exact_answer'][0], list) and len(row['exact_answer']) == 1:
                    dataset['label'].append(row['exact_answer'][0])
                elif isinstance(row['exact_answer'], list) and isinstance(row['exact_answer'][0], str):
                    dataset['label'].append(row['exact_answer'])
                else:
                    raise ValueError(f"unexpected parsing label for {row['id']}: {row['exact_answer']}")
            else:
                raise ValueError(f"Unexpected question type {row['type']}")
            
            dataset["id"].append(row["id"])
            dataset["content"].append(row["body"])
            dataset["type"].append(row["type"])


        assert len(dataset["id"]) == len(dataset["content"]) == len(dataset["label"]), "id content and labels lengths are not the same"
        dataset = datasets.Dataset.from_dict(dataset)
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

def process_APIBench_gorilla(apibench_file):
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

class APIBench_gorilla_HF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_HF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = 'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/huggingface_eval.json'
        api_bench_dataset = process_APIBench_gorilla(apibench_file)
        return api_bench_dataset
    
class APIBench_gorilla_TF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_TF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = 'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/tensorflow_eval.json'
        api_bench_dataset = process_APIBench_gorilla(apibench_file)
        return api_bench_dataset
    
class APIBench_gorilla_TH(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'APIBench_gorilla_TH'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        apibench_file = 'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/apibench/torchhub_eval.json'
        api_bench_dataset = process_APIBench_gorilla(apibench_file)
        return api_bench_dataset


def process_API_gorilla(api_file):
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

class API_gorilla_HF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_HF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """
        api_file = 'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/huggingface_api.jsonl'
        api_dataset = process_API_gorilla(api_file)
        return api_dataset
    

class API_gorilla_TF(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_TF'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """
        api_file = 'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/tensorflowhub_api.jsonl'
        api_dataset = process_API_gorilla(api_file)
        return api_dataset
    
class API_gorilla_TH(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = 'API_gorilla_TH'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        """
        self.split should be one of ['huggingface', 'torchhub', 'tensorflowhub']
        """

        api_file = 'https://raw.githubusercontent.com/ShishirPatil/gorilla/main/data/api/torchhub_api.jsonl'
        api_dataset = process_API_gorilla(api_file)
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
    
def CRB_cat_title_content(x, content_colname="content", title_colname="title"):
    if title_colname is None:
        x["content"] = f"{x[content_colname]}"
    else:
        x["content"] = f"{x[title_colname]}: {x[content_colname]}"
    return x

class CodeRAGBench_programming_solutions(Processor):
    """Contains oracle docs for HumanEval and MBPP"""
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_programming_solutions'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/programming-solutions", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_online_tutorials(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_online_tutorials'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/online-tutorials", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_library_documentation(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_library_documentation'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/library-documentation", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"doc_content", "title_colname":"doc_id"}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_stackoverflow(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_stackoverflow'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/stackoverflow-posts", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset

class CodeRAGBench_gitrepospython(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_gitrepospython'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/github-repos-python", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset
    
class CodeRAGBench_gitrepos(Processor):
    def __init__(self, *args, **kwargs):
        self.dataset_name = 'CodeRAGBench_gitrepos'
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    def process(self):
        dataset = datasets.load_dataset("code-rag-bench/github-repos", num_proc=self.num_proc)[self.split].select(range(100)).map(CRB_cat_title_content, fn_kwargs={"content_colname":"text", "title_colname":None}).select_columns(['content'])
        dataset = dataset.map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        return dataset

class SyllabusQA(Processor):

    def __init__(self, *args, **kwargs):
        self.dataset_name = "SyllabusQA"
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        assert self.split in ['train', 'val', 'test'], "Wrong dataset split, should be one of 'train', 'val', or 'test'."
        url = f"https://raw.githubusercontent.com/umass-ml4ed/SyllabusQA/main/data/dataset_split/{self.split}.csv"
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
        from bs4 import BeautifulSoup
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


class MultiQA(Processor):
    """
    This dataset contains a combination of the following QA datasets:

        < QUERY >                     |     < DOCS >
        nq open                       |     odqa-wiki-corpora-100w-karpukhin ~ kilt-100w
        msmarco 2.1 (first 100k)      |     ms-marco on huggingface = irds/msmarco-passage ~ ms-marco_full on bergen
        adverserial qa                |     squad ~ wikipedia ~ kilt-100w
        hotpotqa                      |     kilt-100w
        wikiqa                        |     kilt-100w
        sciq                          |     kilt-100w
        asqa                          |     kilt-100w
        triviaqa                      |     kilt-100w
        freebase_qa                   |     freebase extract from https://github.com/kelvin-jiang/FreebaseQA (not integrated in bergen)
        squad_v1.1                    |     wikipedia ~ kilt-100w
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'MultiQA'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        ds = datasets.load_dataset("dmrau/multi_qa", num_proc=self.num_proc)[self.split]
        return ds
    

class TechQA(Processor):
    """
    Paper: https://aclanthology.org/2020.acl-main.117.pdf
    Official source: https://github.com/IBM/techqa/tree/master/docker/techqa
    Source we use: https://huggingface.co/datasets/rojagtap/tech-qa
    Note: we combine train/validation/test splits to have a bigger dev dataset
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'TechQA'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        ds = datasets.load_dataset("rojagtap/tech-qa")
        dataset = datasets.concatenate_datasets([ds["train"], ds["validation"], ds["test"]])
        def map_fn(example):
            example['label'] = [example['answer']]
            return example
        dataset = dataset.map(map_fn, num_proc=self.num_proc)
        dataset = dataset.rename_column("question", "content")
        dataset = dataset.remove_columns(["document", "answer"])
        return dataset

class TechQA_docs(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'TechQA_docs'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        ds = datasets.load_dataset("rojagtap/tech-qa")
        dataset = datasets.concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

        def chunk_text(text, title, id, max_size=1000, overlap=200):
            """
            Chunks the given text into parts with a maximum size and overlap, prepending the title to each chunk.
            
            Args:
            - text: The document to chunk
            - title: document title to pre-pend to each chunk
            - id: The id of the document
            - max_size: Maximum size of each chunk
            - overlap: Overlap between adjacent chunks
            
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
                chunks.append({'id': f"{id}_{chunk_id}", 'content': chunk})
                start = end - overlap
                chunk_id += 1

            return chunks

        all_chunks = []
        for i in range(len(dataset)):
            id = dataset[i]["id"]
            doc = dataset[i]["document"]
            assert len(doc.split(' - ')) >= 2
            title, text = doc.split(' - ')[0], ' - '.join(doc.split(' - ')[1:])
            chunks = chunk_text(text, title, id)
            all_chunks.extend(chunks)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_chunks).drop_duplicates(subset='content')).remove_columns(["__index_level_0__"])
        return dataset


class ParaphraseRC(Processor):
    """
    Paper: https://arxiv.org/pdf/1804.07927
    Source: https://huggingface.co/datasets/ibm/duorc/viewer/ParaphraseRC/validation
    DuoRC has two sub datasets: SelfRC (more direct reading comprehension) and ParaphraseRC (more challenging)
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'ParaphraseRC'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def map_fn(row):
            row["content"] = f"{row['title']}: {row['content']}"
            return row
        dataset = datasets.load_dataset("ibm/duorc", "ParaphraseRC")[self.split].filter(lambda x: not x["no_answer"]).rename_columns({"question_id":"id", "question":"content", "answers":"label"}).map(map_fn, num_proc=self.num_proc).remove_columns(["plot_id", "plot", "title", "no_answer"])
        return dataset

class ParaphraseRC_docs(Processor):
    def __init__(self, *args, **kwargs):
        dataset_name = 'ParaphraseRC_docs'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        dataset = datasets.load_dataset("ibm/duorc", "ParaphraseRC")
        if self.split == 'all':
            dataset = datasets.concatenate_datasets([dataset["train"], dataset["validation"], dataset["test"]])
        else:
            dataset = dataset[self.split]

        def chunk_text(text, title, id, max_size=1000, overlap=200):
            """
            Chunks the given text into parts with a maximum size and overlap, prepending the title to each chunk.
            
            Args:
            - text: The document to chunk
            - title: document title to pre-pend to each chunk
            - id: The id of the document
            - max_size: Maximum size of each chunk
            - overlap: Overlap between adjacent chunks
            
            Returns:
            - A list of dictionaries with chunk 'id' and 'content' keys.
            """
            chunks = []
            start = 0
            chunk_id = 0
            while start < len(text):
                end = start + max_size
                if start + overlap > len(text):
                    break
                chunk = text[start:end]
                chunk = title + ": " + chunk  # Prepend the title
                chunks.append({'id': f"{id}_{chunk_id}", 'content': chunk})
                start = end - overlap
                chunk_id += 1

            return chunks
        
        plot_ids = set(dataset["plot_id"])
        plots = {plot_id:None for plot_id in plot_ids}
        all_chunks = []
        for i in tqdm(range(len(dataset))):
            if plots[dataset[i]["plot_id"]] is None:
                id = dataset[i]["plot_id"]
                doc = dataset[i]["plot"]
                title = dataset[i]["title"]
                chunks = chunk_text(doc, title, id)
                all_chunks.extend(chunks)
                plots[dataset[i]["plot_id"]] = True
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_chunks))
        return dataset
    

# problem with this dataset: the questions are not designed for a whole datastore but rather with a fixed given context
class CovidQA(Processor):
    """
    Paper: https://aclanthology.org/2020.nlpcovid19-acl.18/
    Source: https://github.com/deepset-ai/COVID-QA
    HF Source: https://huggingface.co/datasets/deepset/covid_qa_deepset
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'CovidQA'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        def map_fn(row):
            row["label"] = row["answers"]["text"]
            return row
        dataset = datasets.load_dataset("deepset/covid_qa_deepset")[self.split].rename_column("question","content").map(map_fn, num_proc=self.num_proc).remove_columns(["document_id", "context", "is_impossible", "answers"]).cast_column('id', datasets.Value('string'))
        return dataset
    
class CORD19(Processor):
    """
    Paper: https://aclanthology.org/2020.nlpcovid19-acl.1.pdf
    HF Source: https://huggingface.co/datasets/allenai/cord19
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'CORD19'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        dataset = datasets.load_dataset("allenai/cord19", "fulltext", trust_remote_code=True)['train'] # only one split
        all_chunks = []
        for i in tqdm(range(len(dataset))):
            doc = dataset[i]["fulltext"]
            title = dataset[i]["title"]
            chunks = chunk_text(doc, str(i), title, max_size=100, overlap=20, words_or_chars='words')
            all_chunks.extend(chunks)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_chunks))

        return dataset
    
class LoTTE(Processor):
    """
    Source: https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz
    Other possible source: https://huggingface.co/colbertv2
    """
    def __init__(self, path, *args, **kwargs):
        dataset_name = 'LoTTE'
        self.path = path
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        import tarfile
        # download https://downloads.cs.stanford.edu/nlp/data/colbert/colbertv2/lotte.tar.gz in some directory
        parent_dir = os.path.dirname(self.path)
        with tarfile.open(self.path, 'r:gz') as tar:
            if not os.path.exists(os.path.join(parent_dir, 'lotte/')):
                tar.extractall()
                assert os.path.exists(os.path.join(parent_dir, 'lotte/pooled/dev/collection.tsv')), "The extracted directory does not contain the expected files."
        dev_df = pd.read_csv(os.path.join(parent_dir, 'lotte/pooled/dev/collection.tsv'), sep='\t', header=None)
        test_df = pd.read_csv(os.path.join(parent_dir, 'lotte/pooled/test/collection.tsv'), sep='\t', header=None)        
        dev_df['id'] = dev_df[0].apply(lambda x: f"dev_{x}")
        dev_df['content'] = dev_df[1]
        dev_df = dev_df.drop(columns=[0, 1])
        test_df['id'] = test_df[0].apply(lambda x: f"test_{x}")
        test_df['content'] = test_df[1]
        test_df = test_df.drop(columns=[0, 1])
        df = pd.concat([dev_df, test_df])
        all_chunks = []
        for i in range(len(df)):
            chunks = chunk_text(df.iloc[i]['content'], df.iloc[i]['id'], max_size=100, overlap=20, words_or_chars='words')
            all_chunks.extend(chunks)
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(all_chunks))
        return dataset

def process_LoTTE_benchmarks(url: str) -> datasets.Dataset:
    print(f"Downloading {url}...")
    response = requests.get(url)
    response.raise_for_status()
    data = []
    for line in response.text.strip().split('\n'):
        data.append(json.loads(line))
    tmp_df = pd.DataFrame(data=data)
    tmp_df = tmp_df.rename(columns={'qid': 'id', 'question': 'content', 'answer': 'label'})
    dataset = datasets.Dataset.from_pandas(tmp_df)
    dataset = dataset.map(listify_label)
    dataset = dataset.remove_columns([column for column in dataset.column_names if column not in ['id', 'content', 'label']])
    return dataset

class RobustQA_Lifestyle(Processor):
    """
    Paper: RAG-QA Arena https://arxiv.org/pdf/2407.13998
    Source: https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_lifestyle_with_citation.jsonl
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'RobustQA_Lifestyle'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        return process_LoTTE_benchmarks("https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_lifestyle_with_citation.jsonl")
    
class RobustQA_Recreation(Processor):
    """
    Paper: RAG-QA Arena https://arxiv.org/pdf/2407.13998
    Source: https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_recreation_with_citation.jsonl
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'RobustQA_Recreation'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        return process_LoTTE_benchmarks("https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_recreation_with_citation.jsonl")
    
class RobustQA_Science(Processor):
    """
    Paper: RAG-QA Arena https://arxiv.org/pdf/2407.13998
    Source: https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_science_with_citation.jsonl
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'RobustQA_Science'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        return process_LoTTE_benchmarks("https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_science_with_citation.jsonl")
    
class RobustQA_Technology(Processor):
    """
    Paper: RAG-QA Arena https://arxiv.org/pdf/2407.13998
    Source: https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_technology_with_citation.jsonl
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'RobustQA_Technology'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        return process_LoTTE_benchmarks("https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_technology_with_citation.jsonl")
    
class RobustQA_Writing(Processor):
    """
    Paper: RAG-QA Arena https://arxiv.org/pdf/2407.13998
    Source: https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_writing_with_citation.jsonl
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'RobustQA_Writing'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        return process_LoTTE_benchmarks("https://raw.githubusercontent.com/awslabs/rag-qa-arena/refs/heads/main/data/annotations_writing_with_citation.jsonl")
    
class FiQA(Processor):
    """
    Challenge: https://sites.google.com/view/fiqa/
    Source: https://huggingface.co/datasets/LLukas22/fiqa
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'FiQA'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        dataset = datasets.load_dataset("LLukas22/fiqa", num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("answer", "label").rename_column("question", "content").map(lambda _, idx: {"id": str(idx)}, with_indices=True).map(listify_label)
        return dataset

class FiQA_corpus(Processor):
    """
    Source: https://huggingface.co/datasets/BeIR/fiqa
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'FiQA_corpus'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        dataset = datasets.load_dataset("BeIR/fiqa", "corpus", num_proc=self.num_proc)["corpus"]
        dataset = dataset.rename_column("_id", "id").rename_column("text", "content").remove_columns(["title"])
        return dataset
    
class SearchQA(Processor):
    """
    Paper: https://arxiv.org/abs/1704.05179
    Source: https://huggingface.co/datasets/kyunghyuncho/search_qa
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'SearchQA'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        dataset = datasets.load_dataset("kyunghyuncho/search_qa", "train_test_val", trust_remote_code=True)[self.split]
        dataset = dataset.rename_column("answer", "label").rename_column("question", "content").map(listify_label).map(lambda _, idx: {"id": str(idx)}, with_indices=True)
        dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['id', 'content', 'label']])
        return dataset
    
class SearchQA_corpus(Processor):
    """
    Paper: https://arxiv.org/abs/1704.05179
    Source: https://huggingface.co/datasets/kyunghyuncho/search_qa
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'SearchQA_corpus'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        # load train, validation test corpus
        if self.split == 'all':
            train = datasets.load_dataset("kyunghyuncho/search_qa", "train_test_val", trust_remote_code=True)["train"]
            valid = datasets.load_dataset("kyunghyuncho/search_qa", "train_test_val", trust_remote_code=True)["validation"]
            test = datasets.load_dataset("kyunghyuncho/search_qa", "train_test_val", trust_remote_code=True)["test"]
            dataset = datasets.concatenate_datasets([train, valid, test])
        else:
            raise NotImplementedError("Split not implemented")
        all_search_results = []
        all_urls = []
        n_docs = []
        for i in range(len(dataset)):
            snippets = dataset[i]["search_results"]["snippets"]
            n_docs.append(len(snippets))
            urls = dataset[i]["search_results"]["urls"]
            assert len(snippets) == len(urls)
            for j in range(len(snippets)):
                all_search_results.append(snippets[j])
                all_urls.append(urls[j])
        if len(set(all_urls)) == len(all_urls):
            print("There are duplicate URLs in the dataset. Using custom ids.")
            all_urls = [f"{i}" for i in range(len(all_search_results))]
        dataset = datasets.Dataset.from_pandas(pd.DataFrame({"content": all_search_results, "id": all_urls})).filter(lambda x: x['content'] is not None)
        return dataset