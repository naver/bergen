#BERGEN
#Copyright (c) 2024-present NAVER Corp.
#CC BY-NC-SA 4.0 license

import hydra
from multiprocess import set_start_method
import os
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
        
    # # Synchronize all processes after training
    # if dist.is_initialized():
    #     dist.barrier()
        
    # Evaluation only on main process:
    # if int(os.environ.get('LOCAL_RANK', 0)) == 0:
    print(f'Bergen evaluation is running on {config.eval_split} split')
    rag.eval(dataset_split=config.eval_split)

if __name__ == "__main__":
    
    # # First handle the local_rank argument separately with argparse
    # import argparse
    # import sys

    # # Manually parse `local_rank` and capture unknown arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--local_rank', type=int, default=0)  # Capture DeepSpeed local rank
    # args, unknown = parser.parse_known_args()  # Capture unknown args for Hydra to process later

    # # Set local_rank as an environment variable (used by DeepSpeed)
    # os.environ['LOCAL_RANK'] = str(args.local_rank)

    # # Pass the unknown arguments back to Hydra for further processing
    # sys.argv = [sys.argv[0]] + unknown  # Reset sys.argv with only unknown arguments
    
    # needed for multiprocessing to avoid CUDA forked processes error
    # https://huggingface.co/docs/datasets/main/en/process#multiprocessing
    set_start_method("spawn")
    main()
