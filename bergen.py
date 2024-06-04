#BERGEN
#Copyright (c) 2024-present NAVER Corp.
#CC BY-NC-SA 4.0 license

import hydra
from multiprocess import set_start_method
import os
import json
if 'CONFIG' in  os.environ:
    CONFIG = os.environ["CONFIG"]
else:
    CONFIG= 'rag'

@hydra.main(config_path="config", config_name=CONFIG, version_base="1.2")
def main(config):

    from modules.rag import RAG
    rag = RAG(**config, config=config)

    if 'train' in config:
        rag.train()

    rag.eval(dataset_split='dev')

if __name__ == "__main__":
    # needed for multiprocessing to avoid CUDA forked processes error
    # https://huggingface.co/docs/datasets/main/en/process#multiprocessing
    set_start_method("spawn")
    main()
