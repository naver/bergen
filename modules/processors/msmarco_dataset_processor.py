from ..dataset_processor import *
import datasets

class MsMarcoFullDocCollection(Processor):
    """
    MS Marco full docs (instead of passages)
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'ms-marco-docs-v1'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        # load from the ir-dataset HF repo
        hf_name = "irds/msmarco-document"
        dataset = datasets.load_dataset(hf_name, 'docs', num_proc=self.num_proc)  # no need for split?
        dataset = dataset.rename_column("doc_id", "id")
        
        def get_text(example):
            concatenated = example['title'] + " " + example['body']
            return {"content": concatenated}
        
        dataset = dataset.map(get_text, num_proc=self.num_proc)
        dataset = dataset.remove_columns(['url','title', 'body'])
        return dataset

class MsMarcoFullDocQueries(Processor):
    """
    MS Marco queries
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'ms-marco-docs-v1-queries-dev'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        
    def process(self):
       import ir_datasets
       ird = ir_datasets.load("msmarco-document/train")
       Qid = [q.query_id for q in ird.queries_iter()]
       Qtext = [q.text for q in ird.queries_iter()]
       hf_dataset= datasets.Dataset.from_dict({'id':Qid, 'content':Qtext})
       return hf_dataset

class MsMarcoFullDocChunkCollection(Processor):
    """
    MS Marco with our custom random-length passages
    Each document from MS Marco is split into passages of random length
    Used in Provence training: https://arxiv.org/abs/2501.16214
    """
    def __init__(self, *args, **kwargs):
        dataset_name = 'ms-marco-docs-v1-chunked-v1'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)
        
    def process(self):
        # load from the ir-dataset HF repo
        hf_name = "irds/msmarco-document"
        dataset = datasets.load_dataset(hf_name, 'docs', num_proc=self.num_proc)  # no need for split?
        dataset = dataset.rename_column("doc_id", "id")
        
        def my_generator():
            for example in dataset:
                doc_text = example['title'] + " " + example['body']
                sents = doc_text.split("\n")
                # select title
                if sents[0] != "":
                    title = sents[0]
                else:
                    title = ""
                # title will be prepended to each chunk
                sents = [sent for sent in sents[1:] if sent != ""]
                # split into chunks
                chunks = []
                left = 0
                right = 0
                while right < len(sents):
                    l = np.random.rand() * np.random.rand() * 0.99
                    l = 10 - int(l*10) # l \in [1, 10], distribution prioritizes larger numbers
                    right = left + l
                    c = sents[left:right]
                    chunks.append([title]+c)
                    left = right
                did = example['id']
                for ic, c in enumerate(chunks):
                    newdoc_id = did + ':' + str(ic)
                    yield {'id':newdoc_id,'content':' '.join(c)}
        return datasets.Dataset.from_generator(my_generator)