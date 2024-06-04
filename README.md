# BERGEN  Library
We present here a library to benchmark RAG systems, with a focus on Question Answering. Inconsistent benchmarking poses a major challenge in comparing approaches and understanding the impact of each
component in a RAG pipeline.
BERGEN was designed to ease reproducibility and integration of new datasets and models thanks to HuggingFace.



## Quick Start
```
# A RAG setup is  typically a pipeline
# input >> retriever >> reranker >> LLMs >> output
# ideally we want something like that
python3 bergen.py retriever=$retriever reranker=$reranker generator=$LLMS dataset=$dataset 
```
To do so, one can write simple config files (yaml), configuring a retriever, reranker and LLMS for generations. All those configurations can be chained together as follows: am experiment with retrieval using `BM25`, reranking using `MiniLM6`, generation using `tinyllama-chat` in debug mode (using 15 queries) on `kilt_nq`.

```bash
  python3 bergen.py retriever="bm25" reranker="minilm6" generator='tinyllama-chat' dataset='kilt_nq' +debug=True
```

## Overview
BERGEN contains simple wrappers over the following datasets:
 - KILT Benchmkarck: Natural Questions, HotPotQA, Wow,TriviaQA, TRec,..., FEVER
 - POPQA,SCIQ, ASQA,TruthfulQA
- Retriever, Rerankers

All the  configuration files are located in the config dir.
The main config file is located in in config/rag.yaml

```
# main variables locating the local data folder and index
run_name: null
dataset_folder: 'datasets/' # where to dowload,save and preprocess dataset
index_folder: 'indexes/' # where to search index ared saved
runs_folder: 'runs/' # where the text search runs are saved, ie (query and document id lists)
experiments_folder: 'experiments/'    # where the generations from LLMs and metrices are saved

```

For the main command line:  `retriever`, `reranker` and `generator` are optional and can be `None`, the `dataset` argument must always be provided. 
```bash
 
  # no rag setup
  python3 bergen.py  generator='tinyllama-chat' dataset='kilt_nq' +debug=True

  # Retriever - only first stage
  python3 bergen.py retriever="splade-v3" generator='tinyllama-chat' dataset='kilt_nq' +debug=True

  # Retriever Reranker 
  python3 bergen.py retriever="splade-v3" reranker="debertav3"  generator='tinyllama-chat' dataset='kilt_nq' +debug=True

  # using vllm for the generator part with SOLAR
  python3 bergen.py retriever="splade-v3" reranker="debertav3"  generator='vllm_SOLAR-107B' dataset='kilt_nq' +debug=True

```



Datasets will be downloaded, pre-processed, indexed and saved if they do not exist yet, otherwise they will be loaded from `dataset_folder` and `index_folder` respectively. 

```bash
ls config/dataset/
2wikimultihopqa.yaml  kilt_cweb.yaml   kilt_hotpotqa.yaml     kilt_structured_zeroshot.yaml  kilt_wned.yaml  msmarco.yaml  pubmed_bioasq.yaml  ut1.yaml asqa.yaml kilt_eli5.yaml   kilt_nq_wiki2024.yaml  kilt_trex.yaml   kilt_wow.yaml   nq_open.yaml  sciq.yaml  wiki_qa.yaml kilt_aidayago2.yaml   kilt_fever.yaml  kilt_nq.yaml kilt_triviaqa.yaml mmlu.yaml popqa.yaml  truthful_qa.yaml
```


All datasets can be overwritten by adding `+overwrite_datasets=True` as an argument (`Caution`: This might overwrite collections that take long long to encode). In case the indexing is interrupted you can continue encoding a collection from batch 1000 by additionally using the argument `+continue_batch=1000`.

Indexing will be automatically launched if needed: retrieval, reranking runs will be loaded from files if they already exist in `runs`, otherwise they will be created.  Retrieval will only be evaluated if the `query` dataset contains the field `ranking_label`.
For details about indexing, please refer to [indexing.md](documentations/indexing.md)

Experiments are saved under `experiments_folder`. The experiments folder is named after the hash of the config, unless the experiment is finished the folder name will contain the prefix `tmp_`. The script will be aborted if an experiment with the exact same parameters has been run before. To overwrite the experiment add `+overwrite_exp=True` as an argument.

To overwrite an existing index (and subsequently the ranking run) add `+overwrite_index=True` as an argument.

To print the results in a table run. By default this will print all experiments that contain generation metric files in `experiments/` and sort them by the `generator`.

```bash
python3 print_results.py --folder experiments/
TODO Give an example here
```


## Code Structure
```
|-- config
|   |-- dataset/
|   |-- generator/
|   |-- prompt/
|   |-- reranker/
|   |-- retriever/
|   |-- train/
|   |-- rag.yaml
|-- evaluation
|-- eval.py
|-- main.py
|-- models/
|   |-- evaluators/
|   |-- generators/
|   |-- rerankers/
|   |-- retrievers/
|-- modules/
|   |-- dataset_processor.py
|   |-- evaluation.py
|   |-- generate.py
|   |-- rag.py
|   |-- rerank.py
|   |-- retrieve.py
|   `-- utils.py
|-- scripts/
|-- README.md
|-- requirements.py
|-- utils.py

```

## Evaluation
Non-neural metrics will be calculated automatically. Neural metrics such as `BEM` and `LLM` need to be evoked seperately.

By default `eval.py` will scan all folders in `experiments/` and evaluate them sequentially. To evaluate a single folder pass the folder using `--folder`. To avoid running out of memory either run `BEM` using `--bem` or run `LLM` using `--llm`. A csv file will automatically be saved to `results/` containing the table in `csv` format.

```bash
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm
```




## Output files
Example files generated for split `dev` using `naver_splade-cocondenser-selfdistil` as a retriever.
- `config.yaml` The parameters of the experiment in yaml format.
- `eval_dev_generation_time.json` The generation time in json format.
- `eval_dev_metrics.json` Generation evaluation metrics in json format.
- `eval_dev_out.json` Output of the generation, contains `q_id` (str), `response` `(str)` the generated response, `label` `(list (str))` the answer reference (multiple possible), `instruction` `(str)` the instruction given to the generator, `ranking_label` `(list(list(str)), optional)` ids of reference paragraph (again multiple references possible).
- `run.retrieve.top_5.kilt_nq.dev.naver_splade-cocondenser-selfdistil.trec` The retrieval run in `trec` format.
- `eval_dev_ranking_metrics.json` Retrieval evaluation metrics in json format.

### Printing Results Table

Simply run:
```bash
python3 print_results.py --folder experiments/
```
## Extensions
See here our [reference guide](documentations/extensions.md) to add new datasets, models or configure prompts


### Oracle Provenances as Answer
Generating answers using oracle provenances directly as an answer. 

For running the generation simply selectn as the retriever and the generator `oracle_provenance`. For example: 

```python
python3 main.py dataset='kilt_nq' retriever='oracle_provenance' generator='oracle_provenance'
```
# Testing

To run all tests run:

To run all tests in the `tests` folder run:

```bash
pytest tests/
```

To run a single test (e.g. `tinyonly`) run: 
```bash 
pytest tests/ -k "tinyonly"
```

## License
```
BERGEN
Copyright (C) 2024-present NAVER Corp.

Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

A summary of the CC BY-NC-SA 4.0 license is located here:
    https://creativecommons.org/licenses/by-nc-sa/4.0/

The CC BY-NC-SA 4.0 license is located here: 
    https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
```
