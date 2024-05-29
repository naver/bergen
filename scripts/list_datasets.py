import inspect

import sys
sys.path.append('../')
import modules.dataset_processor

datasets =   sys.modules['modules.dataset_processor']
lclasses = inspect.getmembers(datasets, inspect.isclass)

print("Datasets (collection and queries):")
for d in lclasses:
    if d[1].__module__ == 'modules.dataset_processor':
        if d[1].__base__.__name__ == "Processor":
            print(f"{d[0]}") 