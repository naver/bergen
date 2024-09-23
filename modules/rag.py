'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from modules.retrieve import Retrieve
from modules.rerank import Rerank
from modules.generate_query import GenerateQueries
from modules.process_context import ProcessContext
from modules.dataset_processor import ProcessDatasets
from modules.metrics import RAGMetrics
import time 
import shutil
import os 
from tqdm import tqdm
import json
from hydra.utils import instantiate
from utils import (
    eval_retrieval_kilt, init_experiment, move_finished_experiment,
    write_trec, prepare_dataset_from_ids, load_trec,
    print_generate_out, print_rag_model,
    write_generated, write_dict, get_by_id, get_index_path, get_query_generation_filename,
    get_context_processing_filename,
    get_reranking_filename, format_time, get_ranking_filename, get_finished_experiment_name
)

class RAG:
    def __init__(self, 
                generator=None, 
                retriever=None, 
                reranker=None,
                query_generator=None, 
                context_processor=None,
                
                runs_folder=None,
                run_name=None, 
                dataset=None, 
                processing_num_proc=1,
                dataset_folder='datasets/',
                index_folder='indexes/',
                generated_query_folder='generated_queries/',
                processed_context_folder='processed_contexts/',
                experiments_folder='experiments/', 
                qrels_folder='qrels/',
                overwrite_datasets=False,
                overwrite_exp=False,
                overwrite_index=False,
                retrieve_top_k=1,
                rerank_top_k=1,
                generation_top_k=1,
                pyserini_num_threads=1,
                config=None,
                debug=False,
                continue_batch=None,
                train=None,
                prompt=None,
                **kwargs,
                ):
        
        retriever_config = retriever
        reranker_config = reranker
        generator_config = generator
        query_generator_config = query_generator
        context_processor_config = context_processor
        dataset_config = dataset


        #if all the config are still None, load from config

        #if none, then load from config
        if generator_config is None:
            generator_config = config.generator if hasattr(config, 'generator') else None
        if query_generator_config is None:
            query_generator_config = config.query_generator if hasattr(config, 'query_generator') else None
        if retriever_config is None:
            retriever_config = config.retriever if hasattr(config, 'retriever') else None
        if reranker_config is None:
            reranker_config = config.reranker if hasattr(config, 'reranker') else None
        if context_processor_config == None:
            context_processor_config = config.context_processor if hasattr(config, 'context_processor') else None
        if dataset_config == None:
            dataset_config = config.dataset if hasattr(config, 'dataset') else None

        if query_generator_config is None:
            query_generator_config = {"init_args": {"_target_": "models.query_generators.copy.CopyQuery"}}
        
        self.debug = debug
        self.dataset_folder = dataset_folder
        self.experiments_folder = experiments_folder
        self.runs_folder = runs_folder
        self.generated_query_folder = generated_query_folder
        self.processed_context_folder = processed_context_folder
        self.qrels_folder = qrels_folder
        self.run_name = run_name
        self.processing_num_proc = processing_num_proc
        self.index_folder = index_folder
        self.config = config
        self.retrieve_top_k = retrieve_top_k
        self.rerank_top_k = rerank_top_k
        self.generation_top_k = generation_top_k
        self.pyserini_num_threads = pyserini_num_threads
        self.overwrite_exp = overwrite_exp
        self.overwrite_index = overwrite_index
        self.training_config = train

        assert self.generation_top_k <= self.rerank_top_k <= self.retrieve_top_k
        # init experiment (set run name, create dirs)
        self.run_name, self.experiment_folder = init_experiment(config, experiments_folder, index_folder, runs_folder, run_name, overwrite_exp=self.overwrite_exp, continue_batch=continue_batch)
        # process datasets, downloading, loading, covert to format
        self.datasets = ProcessDatasets.process(
            dataset_config, 
            out_folder=self.dataset_folder, 
            num_proc=processing_num_proc,
            overwrite=overwrite_datasets,
            debug=debug,
            shuffle_labels=True if generator_config is not None and generator_config.init_args.model_name == 'random_answer' else False,
            oracle_provenance=True if retriever_config is not None and retriever_config.init_args.model_name == 'oracle_provenance' else False,
            )
        
        self.metrics = {
            "train": RAGMetrics,
            # lookup metric with dataset name (tuple: dataset_name, split) 
            "dev": RAGMetrics, 
            "test": None,
        }

        # init retriever
        self.retriever = Retrieve(
                    **retriever_config,
                    pyserini_num_threads=self.pyserini_num_threads,
                    continue_batch=continue_batch,
                    ) if retriever_config is not None else None
        # init reranker
        self.reranker = Rerank(
            **reranker_config,
            ) if reranker_config is not None else None

        # Hydra way of instantiating generator object defined in config.
        self.generator = instantiate(generator_config.init_args, prompt=prompt) if generator_config is not None else None

        self.query_generator = GenerateQueries(**query_generator_config) if query_generator_config != None else None

        self.context_processor = ProcessContext(**context_processor_config) if context_processor_config != None else None
        
                # print RAG model
        print_rag_model(self, retriever_config, reranker_config, generator_config)
        
    def eval(self, dataset_split):

        dataset = self.datasets[dataset_split]
        query_dataset_name = self.datasets[dataset_split]['query'].name
        doc_dataset_name = self.datasets[dataset_split]['doc'].name

        # query generation (or copying in case query_generator="copy")
        if self.retriever is not None:
            dataset = self.generate_query(
                dataset,
                query_dataset_name, 
                dataset_split, 
            )
        
        # retrieve
        if self.retriever is not None:
            query_ids, doc_ids, _ = self.retrieve(
                    dataset, 
                    query_dataset_name, 
                    doc_dataset_name,
                    dataset_split, 
                    self.retrieve_top_k,
                    )  
        else:
            query_ids, doc_ids = None, None
        # rerank
        if self.reranker is not  None:
            query_ids, doc_ids, _ = self.rerank(
                dataset, 
                query_dataset_name, 
                doc_dataset_name,
                dataset_split, 
                query_ids, 
                doc_ids,
                self.rerank_top_k,
                )
            
        # Now in some cases the retrieval file is larger (because of idx_min, idx_max in dataset, but file was computed on whole dataset)
        # We filter out irrelevant query_ids here:
        filtered_query_ids, filtered_doc_ids = [], []
        for q_id, d_id in zip(query_ids, doc_ids):
            if q_id in dataset['query'].id2index:
                filtered_query_ids += [q_id]
                filtered_doc_ids += [d_id]
                
        query_ids = filtered_query_ids
        doc_ids = filtered_doc_ids
                
        print('Kept:', len(query_ids))
                
        # generate
        if self.generator is not None:
            questions, _, predictions, references = self.generate(
                dataset, 
                dataset_split, 
                query_dataset_name, 
                doc_dataset_name, 
                query_ids, 
                doc_ids,
                )
            # eval metrics
            self.eval_metrics(
                dataset_split, 
                questions, 
                predictions, 
                references
                )

        move_finished_experiment(self.experiment_folder)


    def generate_query(self, dataset, query_dataset_name, dataset_split):
        id2index = dataset['query'].id2index
        if self.query_generator.get_clean_model_name() == "copy":
            dataset['query'] = dataset['query'].add_column("generated_query", dataset['query']["content"])
        else:
            gen_query_file = get_query_generation_filename(
                self.generated_query_folder, 
                query_dataset_name, 
                self.query_generator.get_clean_model_name(), 
                dataset_split
            )
            if not os.path.exists(gen_query_file) or self.overwrite_exp or self.overwrite_index:
                generated_queries = self.query_generator.eval(dataset['query'])
                os.makedirs(self.generated_query_folder, exist_ok=True)
                with open(gen_query_file, 'w') as fp: 
                    json.dump({"generated_queries": generated_queries}, fp)
            else:
                with open(gen_query_file, 'r') as fp: 
                    generated_queries = json.load(fp)["generated_queries"]
            dataset['query'] = dataset['query'].add_column("generated_query", generated_queries)
            shutil.copyfile(gen_query_file, f'{self.experiment_folder}/{gen_query_file.split("/")[-1]}')
        dataset['query'].id2index = id2index
        return dataset
    
    def retrieve(self, 
                 dataset, 
                 query_dataset_name, 
                 doc_dataset_name,
                 dataset_split, 
                 retrieve_top_k,
                 eval_ranking=True,
                 ):
        
        ranking_file = get_ranking_filename(
            self.runs_folder,
            query_dataset_name,
            doc_dataset_name,
            self.retriever.get_clean_model_name(),
            dataset_split, 
            retrieve_top_k,
            self.query_generator.get_clean_model_name()
        )
        #if return_embeddings:
                #raise NotImplementedError('For returning Embeddings is not yet fully implemented!')
        doc_embeds_path = get_index_path(self.index_folder, doc_dataset_name, self.retriever.get_clean_model_name(), 'doc')
        query_embeds_path = get_index_path(self.index_folder, query_dataset_name, self.retriever.get_clean_model_name(), 'query', dataset_split=dataset_split, query_generator_name=self.query_generator.get_clean_model_name())
        if not os.path.exists(ranking_file) or self.overwrite_exp or self.overwrite_index:
            print(f'Run {ranking_file} does not exists, running retrieve...')
             # retrieve
            out_ranking = self.retriever.retrieve(
                dataset,
                query_embeds_path,
                doc_embeds_path,
                retrieve_top_k,
                overwrite_index=self.overwrite_index
                )
            query_ids, doc_ids, scores = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['score']
            write_trec(ranking_file, query_ids, doc_ids, scores)
        else:             
            query_ids, doc_ids, scores = load_trec(ranking_file)
        # copy ranking file to experiment folder    
        shutil.copyfile(ranking_file, f'{self.experiment_folder}/{ranking_file.split("/")[-1]}')
        if eval_ranking:
            if 'ranking_label' in self.datasets[dataset_split]['query'].features:
                print('Evaluating retrieval...')
                wiki_doc_ids = [get_by_id(self.datasets[dataset_split]['doc'], doc_ids_q, 'wikipedia_id') for doc_ids_q in tqdm(doc_ids, desc='Getting wiki ids...')]
                eval_retrieval_kilt(
                    self.experiment_folder, 
                    self.qrels_folder, 
                    query_dataset_name, 
                    doc_dataset_name,
                    dataset_split, query_ids, 
                    wiki_doc_ids, scores, 
                    top_k=self.generation_top_k, 
                    debug=self.debug,
                    )
                
        return query_ids, doc_ids, scores

    def rerank(self, 
               dataset, 
               query_dataset_name, 
               doc_dataset_name, 
               dataset_split, 
               query_ids, 
               doc_ids, 
               rerank_top_k, 
               return_embeddings=False
               ):
        
        doc_ids = [doc_ids_q[:rerank_top_k] for doc_ids_q in doc_ids]

        reranking_file = get_reranking_filename(
            self.runs_folder,
            query_dataset_name,
            doc_dataset_name,
            dataset_split,
            self.retriever.get_clean_model_name(),
            self.retrieve_top_k,
            self.reranker.get_clean_model_name(),
            self.rerank_top_k,
            self.query_generator.get_clean_model_name()
        )

        if not os.path.exists(reranking_file) or self.overwrite_exp:
            rerank_dataset = prepare_dataset_from_ids(
                    dataset, 
                    query_ids, 
                    doc_ids,
                    multi_doc=False,
                    query_field="generated_query"
                )
            out_ranking = self.reranker.eval(rerank_dataset, return_embeddings=return_embeddings)
            query_ids, doc_ids, scores = out_ranking['q_id'], out_ranking['doc_id'], out_ranking['score']
            write_trec(reranking_file, query_ids, doc_ids, scores)
        else:
            # copy reranking file to experiment folder 
            shutil.copyfile(reranking_file, f'{self.experiment_folder}/{reranking_file.split("/")[-1]}')
            query_ids, doc_ids, scores = load_trec(reranking_file)
        if 'ranking_label' in self.datasets[dataset_split]['query'].features:
            print('Evaluating retrieval...')
            wiki_doc_ids = [get_by_id(dataset['doc'], doc_ids_q, 'wikipedia_id') for doc_ids_q in doc_ids]
            eval_retrieval_kilt(
                self.experiment_folder, 
                self.qrels_folder, 
                query_dataset_name, 
                doc_dataset_name,
                dataset_split, 
                query_ids, 
                wiki_doc_ids, 
                scores, 
                top_k=self.generation_top_k, 
                reranking=True, 
                debug=self.debug
                )
            
        return query_ids, doc_ids, scores

    def process_context(self, gen_dataset, 
                       query_dataset_name, 
                       doc_dataset_name, 
                       dataset_split):
        process_context_file = get_context_processing_filename(
            self.processed_context_folder, 
            query_dataset_name,
            doc_dataset_name,
            dataset_split,
            self.retriever.get_clean_model_name(),
            self.retrieve_top_k,
            self.reranker.get_clean_model_name() if self.reranker is not None else None,
            self.rerank_top_k,
            self.query_generator.get_clean_model_name(),
            self.context_processor.get_clean_model_name(),
        )
        if not os.path.exists(process_context_file) or self.overwrite_exp or self.overwrite_index:
            processed_contexts = self.context_processor.eval(gen_dataset['doc'], gen_dataset['query'])
            os.makedirs(self.processed_context_folder, exist_ok=True)
            with open(process_context_file, 'w') as fp: 
                json.dump({"processed_contexts": processed_contexts}, fp)
        else:
            with open(process_context_file, 'r') as fp: 
                processed_contexts = json.load(fp)["processed_contexts"]
        #gen_dataset_new = gen_dataset.map(lambda ex: {"doc": processed_contexts}, batched=True)
        gen_dataset = gen_dataset.remove_columns('doc')
        gen_dataset = gen_dataset.add_column('doc', processed_contexts)
        shutil.copyfile(process_context_file, f'{self.experiment_folder}/{process_context_file.split("/")[-1]}')
        return gen_dataset
    
    def generate(self, 
                 dataset, 
                 dataset_split, 
                 query_dataset_name, 
                 doc_dataset_name, 
                 query_ids, 
                 doc_ids,
                 ):
        doc_ids = [doc_ids_q[:self.generation_top_k] for doc_ids_q in doc_ids] if doc_ids is not None else doc_ids 

        gen_dataset = prepare_dataset_from_ids(
            dataset, 
            query_ids, 
            doc_ids,
            multi_doc=True, 
            query_field="content",
            )

        if self.context_processor is not None and self.retriever is not None:
            gen_dataset = self.process_context(
                                               gen_dataset, 
                                               query_dataset_name, 
                                               doc_dataset_name, 
                                               dataset_split)
        
        generation_start = time.time()
        query_ids, questions, instructions, predictions, references, ranking_labels  = self.generator.eval(gen_dataset)
        generation_time = time.time() - generation_start
        write_generated(
            self.experiment_folder,
            f"eval_{dataset_split}_out.json",
            query_ids, 
            questions,
            instructions, 
            predictions, 
            references, 
            ranking_labels
        )

        print_generate_out(
            questions,
            instructions,
            predictions,
            query_ids, 
            references,
            ranking_labels,
            )

        
        if hasattr(self.generator,"total_cost"):
            print(self.generator.total_cost,self.generator.prompt_cost, self.generator.completion_cost)
            write_dict(self.experiment_folder, f"eval_{dataset_split}_generation_cost.json", 
                       {'total_cost':self.generator.total_cost,
                        'prompt_cost':self.generator.prompt_cost,
                        'completion_cost':self.generator.completion_cost}
                        )


        formated_time_dict = format_time("Generation time", generation_time)
        write_dict(self.experiment_folder, f"eval_{dataset_split}_generation_time.json", formated_time_dict)

        return questions, instructions, predictions, references

    def eval_metrics(self, dataset_split, questions, predictions, references):
        if predictions == references == questions == None:
            return
        metrics_out = self.metrics[dataset_split].compute(
        predictions=predictions, 
        references=references, 
        questions=questions
        )
        write_dict(self.experiment_folder, f"eval_{dataset_split}_metrics.json", metrics_out)
    

    def train(self):
        from transformers import TrainingArguments
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from modules.trainer import RAGTrainer
        from modules.dataset import Tokenized_Sorted_Dataset
        from accelerate import Accelerator

        dataset_split = 'train'
        dataset = self.datasets[dataset_split] 
        query_dataset_name = dataset['query'].name
        doc_dataset_name = dataset['doc'].name
        
        if self.retriever is not None:
            dataset = self.generate_query(dataset, query_dataset_name, dataset_split)

        # if no retriever don't load doc embeddings
        if self.retriever is not None:
            query_ids, doc_ids, _ = self.retrieve(
                dataset, 
                query_dataset_name, 
                doc_dataset_name,
                dataset_split, 
                self.retrieve_top_k,
                eval_ranking=False
                )            
        else:
            query_ids, doc_ids = None, None

        if self.reranker is not None:
            query_ids, doc_ids, _ = self.rerank(
                dataset,
                query_dataset_name,
                doc_dataset_name,
                dataset_split,
                query_ids,
                doc_ids,
                self.rerank_top_k,
                )
            
        # get top-k docs
        doc_ids = [doc_ids_q[:self.generation_top_k] for doc_ids_q in doc_ids] if doc_ids is not None else doc_ids
        
        # prepare dataset
        gen_dataset = prepare_dataset_from_ids(
            dataset, 
            query_ids, 
            doc_ids, 
            multi_doc=True, 
            )

        # context processing if needed
        if self.context_processor is not None and self.retriever is not None:
            gen_dataset = self.process_context(
                                               gen_dataset, 
                                               query_dataset_name, 
                                               doc_dataset_name, 
                                               dataset_split)
        
        # split train into train and test
        if isinstance(self.training_config.test_size_ratio, int):
            self.training_config.test_size_ratio = min(len(gen_dataset)//2, self.training_config.test_size_ratio)
            
        train_test_datasets = gen_dataset.train_test_split(self.training_config.test_size_ratio, seed=42)
        
        print("Preprocessing data...")
        from models.generators.llm_cocom import LLMCocom
        if not isinstance(self.generator, LLMCocom):
            train_test_datasets['train'] = Tokenized_Sorted_Dataset(train_test_datasets['train'], self.generator, training=True)
            train_test_datasets['test'] = Tokenized_Sorted_Dataset(train_test_datasets['test'], self.generator, training=True)
        else:
            # Preprocessing for COCOM is all done in the collate_fn function.
            self.generator.model.generation_top_k = self.generation_top_k
            
            # While we are at it we add the "sep" ! 
            self.generator.model.sep = True
            self.generator.model.config.sep = True # also in the config so that it'll be saved (probably!)
            
            if  self.training_config.get("freeze_compressor", False):
                kept_adapters = [elt for elt in self.generator.model.adapter_keys if elt != 'encoder_adapter']
                print('Freezing the compressor, keeping only', kept_adapters)
                self.generator.model.decoder.set_adapter(kept_adapters)
                
            # We need to re-activate the gradients for these tokens if needed
            self.generator.model.prepare_mem_tokens_optimization()
                            
        print("Data preprocessed")

        # if lora in train config
        if 'lora' in self.training_config:
            self.generator.model = prepare_model_for_kbit_training(self.generator.model)
            print("using lora training")
            # lora config
            lora_config = LoraConfig(
                **self.training_config.lora,
                target_modules='all-linear',
                )
            # get adapter
            self.generator.model = get_peft_model(self.generator.model, lora_config)
            self.generator.model.print_trainable_parameters()
            self.generator.model = self.generator.model.bfloat16()
            
        # TODO: for other models, seems like '.eval' was called, we should do .train()            
        model = self.generator.model
        accelerator = Accelerator(split_batches=False)

        args = TrainingArguments(
            run_name=self.run_name,
            output_dir=f'{self.experiment_folder}/train/',
            **self.training_config.trainer,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy='steps',
            save_steps=500,
            save_total_limit=20,
            logging_strategy='steps',
            logging_steps=100,
            remove_unused_columns=False,
        )

        trainer = RAGTrainer(
            model=model,
            args=args,
            data_collator=self.generator.collate_fn,
            train_dataset=train_test_datasets['train'],
            eval_dataset=train_test_datasets['test']
        )
        
        model, _, _, _ = accelerator.prepare(
            model, 
            trainer.optimizer, 
            trainer.get_train_dataloader(), 
            trainer.get_eval_dataloader()
        )
        
        trainer.train(resume_from_checkpoint=None)
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)

        self.generator.model = unwrapped_model
        
        if accelerator.is_main_process:
            model.save_pretrained(f"{self.experiment_folder}/last_model")
            move_finished_experiment(self.experiment_folder)
            self.experiment_folder = get_finished_experiment_name(self.experiment_folder)
