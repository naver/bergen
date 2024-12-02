from ..dataset_processor import *
import datasets
import os
from collections import defaultdict
import urllib.request
import json
import pdb
class AmbigQA_UB(Processor):

    def __init__(self, *args, **kwargs):
        dataset_name = 'ambig_qa_ub'
        super().__init__(*args, **kwargs, dataset_name=dataset_name)

    def process(self):
        #dataset = datasets.load_dataset(hf_name, hf_query_or_doc_name, num_proc=self.num_proc)[self.split]
        #ds = load_dataset("sewon/ambig_qa", "light",num_proc=self.num_proc)['train']
        def my_gen():
            sid='ambigqa_ub'
            i=0
            #for split in ['train','dev','test']:
            for split in ['dev','test']:
                ds = load_dataset('erbacher/AmbigNQ-clarifying-question',num_proc=self.num_proc)[split]
                for x in ds:
                    qs=eval(x['intent'])
                    answers=eval(x['answer'])

                    if x['ambig'] is True:
                        #pdb.set_trace()
                        for q,a in zip(qs,answers):
                            qid=sid+str(i)
                            i+=1
                            #print(q,a)
                            if q is not None and a is not None:
                                yield {'id':qid, 'content':q, 'label':[a]}

        
        return datasets.Dataset.from_generator(my_gen)



from datasets import load_dataset
