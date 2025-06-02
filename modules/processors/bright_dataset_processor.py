from ..dataset_processor import *
import datasets
import requests  




   
class BRIGHTDocProcessor(Processor):
    def __init__(self, longdoc,split,*args, **kwargs):
        dataset_name = 'BRIGHT_%s'%split
        super().__init__(*args, **kwargs, split=split,dataset_name=dataset_name)
        self.longdoc = longdoc

    def process(self):
        hf_name = 'xlangai/BRIGHT' 
        doc = 'long_documents' if self.longdoc else 'documents'
        dataset = datasets.load_dataset(hf_name, doc,num_proc=self.num_proc)[self.split]
        return dataset


   
   
class BRIGHTQueryProcessor(Processor):
    def __init__(self, longdoc,split,qlen,*args, **kwargs):
        dataset_name = 'BRIGHTQuery_%s'%split
        super().__init__(*args, **kwargs, split=split,dataset_name=dataset_name)
        self.longdoc = longdoc
        self.qlen = qlen

    def process(self):
        hf_name = 'xlangai/BRIGHT' 
        dataset = datasets.load_dataset(hf_name, "examples",num_proc=self.num_proc)[self.split]
        dataset = dataset.rename_column("query", "content")
        dataset['content'] = dataset['content'].split()[:qlen]
        if self.longdoc:
            dataset = dataset.rename_column("gold_ids_long", "ranking_label")
        else:
            dataset = dataset.rename_column("gold_ids", "ranking_label")

        dataset = dataset.remove_columns(['reasoning', 'excluded_ids','gold_ids_long'])
        
        return dataset


   
