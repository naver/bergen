import datasets
import os
from ..dataset_processor import *


class NIHJsonDataset(Processor):

    def __init__(self, filepath,index=0, *args, **kwargs):
        self.filepath=filepath
        self.dataset_name = os.path.basename(filepath).split('.')[0]
        self.index=index # index to extract query are 0, documents are 1
        
        super().__init__(*args, **kwargs, dataset_name=self.dataset_name)
    
    def process(self):
        X=json.load(open(self.filepath))
        #
        data=[x[self.index] for x in X]
        if self.index==0:
            #reprocess single str label into a list of str
            for y in data:
                y['label']=[y['label']]
        #oracle data
        qids_dids_l =[ (x[0]['id'],x[1]['id']) for x in X]
        #FIXME get the run dirs
        fname='runs/run.oracle.'+self.dataset_name+'.dev.trec'
        #6915606477668963399	q0	10593264_2	0	100	run
        f=open(fname,'w')
        for i in qids_dids_l:
            f.write(i[0]+'\tq0\t'+i[1]+'\t0\t100\trun\n')
        f.close()
        return datasets.Dataset.from_list(data)

#FIXME upload dataset on HF easier 
#Do the 3 configs done reclean the processor
    
    
