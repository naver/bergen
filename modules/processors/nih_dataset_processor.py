import datasets
import os
from ..dataset_processor import *

class NIHDataset(Processor):

    def __init__(self, config_name='number',is_query=True,*args, **kwargs):
        self.dataset_name='nih_v1_'+config_name
        if is_query:
            self.dataset_name+='_query'
        else:
            self.dataset_name+='_doc'
        self.conf_name= config_name
        self.is_query=is_query
        super().__init__(dataset_name=self.dataset_name,*args, **kwargs)
    
    def process(self):
        print(self.conf_name)
        d = datasets.load_dataset('naver/bergen_nih_v1',self.conf_name)[self.split]
        if self.is_query:
            #reprocess single str label into a list of str    
            dataset = d.rename_column("qid", "id")
            dataset = dataset.rename_column("query", "content")
            dataset = dataset.remove_columns(['did', 'doc'])
            dataset = dataset.map(lambda x: {"label": [str(x["label"])]})
            return dataset
        else:
            #document dataset
            dataset = d.rename_column("did", "id")
            dataset = dataset.rename_column("doc", "content")
            dataset = dataset.remove_columns(['qid', 'query','label'])
            print(dataset[:10])
            return dataset

class NIHDatasetNumber(NIHDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,config_name='number')

class NIHDatasetSimple(NIHDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,config_name='simple')

class NIHDatasetMultiHop(NIHDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,config_name='multihop')

    
