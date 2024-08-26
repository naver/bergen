from ..dataset_processor import *
import datasets
import requests  


#FIXME --> this is not used 
class TimeSensitiveQA(Processor):

    def __init__(self,  *args, **kwargs):
        self.dataset_name = 'TimeSensitiveQA'
        #self.path = data_path
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        hf_name = "diwank/time-sensitive-qa"
        #dataset = datasets.load_dataset("json", data_files=[self.path])[self.split]
        dataset = datasets.load_dataset(hf_name, num_proc=self.num_proc)[self.split]
        #['id','docs','question','type','ideal_answer','exact_answer','snippets']
        dataset = dataset.map(lambda example: {'label': example['targets']})
        dataset = dataset.rename_column("question", "content")
        dataset = dataset.rename_column("idx", "id")
        dataset = dataset.remove_columns(['context', 'paragraphs'])
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
        
        ###dataset = dataset.map(lambda example: {'label': [example['correct_answer']]})
        #dataset = dataset.remove_columns(["correct_answer","distractor1", "distractor2","distractor3","support"])
        #generating the id, train_0 ... validation_0 validation_1
        cid= [self.split+str(i) for i in range(len(dataset))]
        dataset = dataset.add_column("id", cid)
        if self.oracle_provenance:
            # document
            dataset = dataset.rename_column('support','content')
            dataset = dataset.remove_columns(["question","correct_answer","distractor1", "distractor2","distractor3"])            
            #dataset = datasets.Dataset.from_dict({'content': paragraphs, 'id': ids})
        else:
            # query
            dataset = dataset.rename_column("question", "content")
            dataset = dataset.map(lambda example: {'label': [example['correct_answer']]})
            dataset = dataset.remove_columns(["support","correct_answer","distractor1", "distractor2","distractor3"])        
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
   
