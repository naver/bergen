import inspect

import sys
sys.path.append('../')
import modules.dataset_processor
import modules.processors.kilt_dataset_processor
import modules.processors.qa_dataset_processor
import modules.processors.mrag_dataset_processor



for dpath in ['modules.dataset_processor','modules.processors.kilt_dataset_processor','modules.processors.qa_dataset_processor','modules.processors.mrag_dataset_processor']:
    datasets =   sys.modules[dpath]

    lclasses = inspect.getmembers(datasets, inspect.isclass)
    #FIXME
    print("Datasets (collection and queries):")
    for d in lclasses:
        if d[1].__module__ == dpath:
            if d[1].__base__.__name__ == "Processor":
                print(f"{d[0]}") 