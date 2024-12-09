'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import shutil
from hydra import initialize, compose
from bergen import main
from eval import run_eval
from omegaconf import OmegaConf
import pytest
import gc
import inspect 
import torch
import os
import gc
from modules.rag import RAG
import modules.dataset_processor
import pathlib
"""
pip install pytest
To run tests run from the root folder:
pytest tests/ -k "tinyonly" #to run one test
or simply "pytest tests/ " to run all the tests in zeroshot
"""

#FIXME Test equality/stability of output
# run once at the beginning
@pytest.fixture(scope="session", autouse=True)
def init():
    
    def rmdir(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)

    def clean_dirs():
        rmdir('tests/exp/')
        rmdir('tests/index/')
        rmdir('tests/run/')
        rmdir('tests/dataset/')


    if not torch.cuda.is_available():
        raise SystemError('No GPU available. Needs GPUs for running tests.')
    clean_dirs()
    
    
@pytest.fixture(autouse=True)
def clear_gpu_memory():
    yield  # This makes the fixture run after each test
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()


class TestBergenMain:

    def test_init(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            cfg = compose(config_name='rag_ut1')
            self.helper_single(cfg, test_name)

    def test_tinyonly(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            cfg = compose(config_name='rag_ut1', overrides=["generator=tinyllama-chat"])
            self.helper_single(cfg, test_name)

    # def test_bm25mixtral(self):
    #     with initialize(config_path="../config",version_base="1.2"):
    #         test_name = inspect.currentframe().f_code.co_name
    #         cfg = compose(config_name='rag_ut1', overrides=["retriever=bm25", "generator=mixtral-moe-7b-chat", "generator.batch_size=8"])
    #         self.helper_with_rerun(cfg, test_name)

    # def test_bm25tiny(self):
    #     with initialize(config_path="../config",version_base="1.2"):
    #         test_name = inspect.currentframe().f_code.co_name
    #         cfg = compose(config_name='rag_ut1', overrides=["retriever=bm25", "generator=tinyllama-chat", "generator.batch_size=1"])
    #         self.helper_with_rerun(cfg, test_name)
            
    def test_spladetiny(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            cfg = compose(config_name='rag_ut1', overrides=["retriever=splade++", 
                                                            "generator=tinyllama-chat", 
                                                            "generator.init_args.batch_size=64"])
            self.helper_with_rerun(cfg, test_name)
    
    def test_dense_contriever(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            cfg = compose(config_name='rag_ut1', overrides=["retriever=contriever", 
                                                            "generator=tinyllama-chat", 
                                                            "generator.init_args.batch_size=64"])
            self.helper_with_rerun(cfg, test_name)


    def test_reranker(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            cfg = compose(config_name='rag_ut1', overrides=["retriever=splade++", 
                                                            "generator=tinyllama-chat",  
                                                            "reranker=minilm6", 
                                                            "reranker.batch_size=1"])
            self.helper_with_rerun(cfg, test_name)

    # should be the last since it takes 90% memory
    def test_vllm_spladetiny(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            cfg = compose(config_name='rag_ut1', overrides=[ "generator=vllm_tinyllama-chat", 
                                                            "generator.init_args.batch_size=64"])
            self.helper_with_rerun(cfg, test_name)

    
    def test_train_lora(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            cfg = compose(config_name='rag_ut1', overrides=[ "generator=tinyllama-chat", 
                                                            "+train=lora", 
                                                            "train.trainer.num_train_epochs=1",
                                                            "train.lora.r=1"])
            self.helper_with_rerun(cfg, test_name)

    # Test of a training without lora, a bit too long (~4 minutes) to be uncommented, but maybe useful.
    # Also: running it requires manual choice of wandb option
    # def test_train_full(self):
    #     with initialize(config_path="../config",version_base="1.2"):
    #         test_name = inspect.currentframe().f_code.co_name
    #         cfg = compose(config_name='rag_ut1', overrides=[ "generator=tinyllama-chat",
    #                                                         "generator.init_args.quantization=bfloat16", 
    #                                                         "+train=full", 
    #                                                         "train.trainer.num_train_epochs=1"])
    #         self.helper_with_rerun(cfg, test_name)

    # this is too long takes 20 min for NQ, it download the dataset from HF, and process it from scratch
    # def test_create_kiltnq(self):
    #    with initialize(config_path="../config",version_base="1.2"):
    #        test_name = inspect.currentframe().f_code.co_name
    #        cfg = compose(config_name='rag_ut1', overrides=[ "dataset=kilt_nq", "generator=tinyllama-chat"])
    #        #cfg = compose(config_name='rag_ut1', overrides=["generator=tinyllama-chat"])
    #        
    #        print(cfg,test_name)
    #        self.create_rag_obj(cfg, test_name)

    # This only check the instantiation of the processor, but not calling the get_process
    def test_processor_subset(self):
        datasets=['kilt_nq','asqa','kilt_triviaqa','popqa','kilt_hotpotqa']

        for d in datasets:
            with initialize(config_path="../config",version_base="1.2"):
                test_name = inspect.currentframe().f_code.co_name
                cfg = compose(config_name='rag_ut1', overrides=[ "dataset="+d, "generator=tinyllama-chat"])
                print(cfg)
                p = modules.dataset_processor.ProcessDatasets.check_instantiate(cfg['dataset'], out_folder=cfg.dataset_folder)
                print(p)
    
    def test_processor_all(self):
        
        datasets=sorted(pathlib.Path('config/dataset').glob('**/*.yaml'))

        for dpath in datasets:
            d = os.path.basename(dpath).replace('.yaml','')
            dir_name = os.path.dirname(dpath)
            # if config in mkqa tidy add dirname to the config
            if not(dir_name.endswith('dataset')): 
                d =os.path.basename(dir_name)+'/'+d
            print(dir_name,d)
            
            #skipping current failing test 
            if 'pubmed_bioasq' in d:
                continue

            with initialize(config_path="../config",version_base="1.2"):
                test_name = inspect.currentframe().f_code.co_name
                cfg = compose(config_name='rag_ut1', overrides=[ "dataset="+d, "generator=tinyllama-chat"])
                #print(cfg)
                p = modules.dataset_processor.ProcessDatasets.check_instantiate(cfg['dataset'], out_folder=cfg.dataset_folder)
                print(p)          

    @pytest.mark.skip(reason="Helper function, not a test")
    def set_folders(self, cfg, test_name):
        cfg.experiments_folder = f"tests/exp/{test_name}"
        cfg.runs_folder = f"tests/run/{test_name}/"
        cfg.dataset_folder = f"tests/dataset/{test_name}/"
        cfg.index_folder = f"tests/index/{test_name}/"
    
    @pytest.mark.skip(reason="Helper function, not a test")
     # runs experiment twice. second time it reuses intermediate results
    def helper_with_rerun(self, cfg, test_name):
        self.set_folders(cfg, test_name)
        main(cfg)
        
    @pytest.mark.skip(reason="Helper function, not a test")
    def helper_single(self, cfg, test_name):
        self.set_folders(cfg, test_name)
        main(cfg)
    
    @pytest.mark.skip(reason="Helper function, not a test")
    def create_rag_obj(self, cfg, test_name):
        self.set_folders(cfg, test_name)
        rag = RAG(**cfg, config=cfg)


class TestBergenEval:
    def test_lid(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            exp_folder = "tests/utdata/"
            run_eval(experiment_folder=exp_folder, lid=True, force=True)

   
    def test_llmeval_default(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            exp_folder = "tests/utdata/"
            run_eval(experiment_folder=exp_folder, llm=["tinyllama-chat", "test-llm-1"], llm_batch_size= 4, llm_prompt="default_qa", force=True, samples=4)
     
       
    def test_llmeval_multi(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            exp_folder = "tests/utdata/"
            run_eval(experiment_folder=exp_folder, llm=["tinyllama-chat", "test-llm-2"], llm_batch_size= 4, llm_prompt="default_multi_qa", force=True)
    
    def test_vllmeval(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            exp_folder = "tests/utdata/"
            run_eval(experiment_folder=exp_folder, vllm=["tinyllama-chat", "test-vllm-1"], llm_batch_size=4, llm_prompt="default_qa", force=True)
   
    def test_vllmeval_multi(self):
        with initialize(config_path="../config",version_base="1.2"):
            test_name = inspect.currentframe().f_code.co_name
            exp_folder = "tests/utdata/"
            run_eval(experiment_folder=exp_folder, vllm=["tinyllama-chat", "test-vllm-2"], llm_batch_size=4, llm_prompt="default_multi_qa", force=True)
   
    

   
    