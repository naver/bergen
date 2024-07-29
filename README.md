# BERGEN: Benchmarking RAG
We present BERGEN (BEnchmarking Retrieval-augmented GENeration), a library to benchmark RAG systems, focusing on question-answering (QA). Inconsistent benchmarking poses a major challenge in comparing approaches and understanding the impact of each component in a RAG pipeline.
BERGEN was designed to ease the reproducibility and integration of new datasets and models thanks to HuggingFace.

For more information and experimental findings, please see:
- the initial BERGEN paper: https://arxiv.org/abs/2407.01102
- the Multilingual RAG paper:  https://arxiv.org/abs/2407.01463

## Quick Start
A RAG setup is typically a pipeline

`question` >> `retriever` >> `reranker` >> `LLM` >> `answer`

One can write simple config files (yaml), configuring a retriever, reranker, and LLMs for generations. All those configurations can be chained together as follows: am experiment with retrieval using `BM25`, reranking using `MiniLM6`, and generation using `tinyllama-chat` on `kilt_nq`.

```bash
  python3 bergen.py retriever="bm25" reranker="minilm6" generator='tinyllama-chat' dataset='kilt_nq'
```

### Installation
 Check  the [installation guide](documentations/INSTALL.md)

## Supported Features
BERGEN contains simple wrappers for the following features:
| Category        | Name               |   Argument                     |
|-|-|-|
| **Datasets**    | NQ                 | `dataset="kilt_nq"`            |
|                 | TriviaQA           | `dataset="kilt_triviaqa"`      |
| **Generators**  | Llama 2 7B Chat    | `generator="llama-2-7b-chat"`  |
|                 | Llama 2 13B Chat   | `generator="llama-2-13b-chat"` |
|                 | Llama 2 70B Chat   | `generator="llama-2-70b-chat"` |
|                 | Mixtral-8x7B-Instruct-v0.1   | `generator="mixtral-moe-7b-chat"` |
|                 | SOLAR-10.7B-Instruct-v1.0   | `generator="SOLAR-107B"` |
| **Retrievers**  | BM25               | `retriever="bm25"`             |
|                 | SPLADE-v3          | `retriever="spladev3"`         |
|                 | BGE                | `retriever="bge"`              |
| **Rerankers**   | DeBERTa-v3         | `reranker="debertav3"`         |

Supported Metrics:
| Metric |
| - |
|Match|
|Exact Match|
|Recall|
|Precision|
| F1 Score |
| ROUGE-1,2,L  |  
|LLMeval|
|Language ID|  

All the  configuration files are located in the config dir.
The main config file is located in config/rag.yaml

```
# main variables locating the local data folder and index
run_name: null
dataset_folder: 'datasets/' # where to download, save and preprocess the dataset
index_folder: 'indexes/' # where the search index are saved
runs_folder: 'runs/' # where the text search runs are saved, ie (query and document id lists)
experiments_folder: 'experiments/'    # where the generations from LLMs and metrics are saved

```

### Datasets
Datasets will be downloaded, pre-processed, indexed, and saved if they do not exist yet, otherwise, they will be loaded from `dataset_folder` and `index_folder` respectively. 

```bash
ls config/dataset/
2wikimultihopqa.yaml  kilt_cweb.yaml   kilt_hotpotqa.yaml     kilt_structured_zeroshot.yaml  kilt_wned.yaml  msmarco.yaml  pubmed_bioasq.yaml  ut1.yaml asqa.yaml kilt_eli5.yaml   kilt_nq_wiki2024.yaml  kilt_trex.yaml   kilt_wow.yaml   nq_open.yaml  sciq.yaml  wiki_qa.yaml kilt_aidayago2.yaml   kilt_fever.yaml  kilt_nq.yaml kilt_triviaqa.yaml mmlu.yaml popqa.yaml  truthful_qa.yaml
```

To add a new datasets, please refer to an following guide:[extensions](documentations/extensions.md)

### Retrieval
Indexing of the document collections will be automatically launched if needed: retrieval, reranking runs will be loaded from files if they already exist in `runs`, otherwise they will be created.  Retrieval will only be evaluated if the `query` dataset contains the field `ranking_label`.
For details about indexing, please refer to [indexing.md](documentations/indexing.md)


Experiments are saved under `experiments_folder`. The experiment folder is named after the hash of the config, unless the experiment is finished the folder name will contain the prefix `tmp_`. You can provide a custom name for the experiment folder by adding `+run_name={YOUR_NAME}`. The script will be aborted if an experiment with the exact same parameters has been run before. To overwrite the experiment add `+overwrite_exp=True` as an argument.


- To overwrite the experiment add `+overwrite_exp=True` as an argument, due to a bug or another update in the config 
- To overwrite an existing retrieval run, `+overwrite_run=True` as an argument.
-  To rebuild the index (and subsequently the ranking run) add `+overwrite_index=True` as an argument.

To print the results in a table run the following commands. By default, this will print all experiments that contain generation metric files in `experiments/` and sort them by the `generator`.

```bash
# will print a markdown of the results and save a csv file under the results directory
python3 print_results.py --csv --folder experiments/

#csv files with all the metrics
exp_folder,Retriever,P_1,Reranker,Generator,gen_time,query_dataset,r_top,rr_top,M,EM,F1,P,R,Rg-1,Rg-2,Rg-L,BEM,LLMeval
216567b3d48ef3fc,naver/splade-v3/,,naver/trecdl22-crossencoder-debertav3,TinyLlama/TinyLlama-1.1B-Chat-v1.0,00:03:53.19,KILTTriviaqa,100,100,0.6763772175536882,0.00018674136321195143,0.11749967712256401,0.07122756370055569,0.5380933823321367,0.1505780809175042,0.055962386132169924,0.14611799602749245,0.47356051206588745,
```
## Pipeline Examples
For the main command line:  `retriever`, `reranker`, and `generator` are optional and can be `None`, the `dataset` argument must always be provided. 
 
Generation without Retrieval (Closed Book)
```bash
python3 bergen.py  generator='tinyllama-chat' dataset='kilt_nq' 
```
Retriever - only first stage:
```bash
python3 bergen.py retriever="splade-v3" generator='tinyllama-chat' dataset='kilt_nq'
```
Retriever + Reranker
```bash
python3 bergen.py retriever="splade-v3" reranker="debertav3"  generator='tinyllama-chat' dataset='kilt_nq'
```

Using vllm to speed up generation:
```bash
python3 bergen.py retriever="splade-v3" reranker="debertav3"  generator='vllm_SOLAR-107B' dataset='kilt_nq'
```

To specify another config file:
```bash
# create a config file located in the config dir
# (the default config is rag)
CONFIG=myownconfig python3 bergen.py retriever="splade-v3" reranker="debertav3"  generator='vllm_SOLAR-107B' dataset='kilt_nq'
```

## Evaluation
Non-neural metrics will be calculated automatically. Neural metrics such as `BEM` and `LLM` need to be evoked seperately.

By default `eval.py` will scan all folders in `experiments/` and evaluate them sequentially. To evaluate a single folder pass the folder using `--folder`. To avoid running out of memory either run `BEM` using `--bem` or run `LLM` using `--llm` or `--vllm` (for faster inference). A csv file will automatically be saved to `results/` containing the table in `csv` format.

When using `--llm` you have a choice on how you transform LLM predictions in the final score:
- direcly check in the generated answer for the expepected label occurence (default Yes/No), and assign corresponding score (default 1/0), when no expected label is found, or more than one expected label is matched, we assign score -100 to the corresponding sample, such samples are excluded from the mean score computation
- rely on the logits assigned to the first token: get values corresponding to the expected labels, normalize them to 1 (get probability distribution across possible labels `p(label)`); final score would correspond to Inline equation: $\sum_{label} score(label)*p(label)$ 
The choice of score interpretation is done via `use_logits` parameter specified at evaluation config file. Default value is set to `True` (corresponding to the second option)

In case of `--vllm` inference point only option 1 is possible. 

```bash
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --vllm
```
Similarly to  `--generator` you can specify which LLM you are willing as first options of `--llm`/`-vllm`, as well as short name at metrics naming (use the name of the configuration file as the name of the llm). 
 

```bash
# use llama2-7b-chat to run evaluation, output metric will be named VLLMeval_l2_7b
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --vllm  "llama-2-7b-chat" "l2_7b"

# use tinyllama to run evaluation, output metric will be named LLMeval_tinyllama
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --llm  "tinyllama-chat" "tinyllama"

# in default settings (with no arguments specified) we use SOLAR-107B for evaluation and output metric is named VLLMeval
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --vllm  

```

You can specify prompt and other parameters in the evaluation config file for `--llm` or `--vllm` at `config/evaluator` directory. By default they rely on `default_qa.yaml` configuration which assigns binary (Yes/No) value to each triple of <em>Question/Response/Gold Response</em>. You can specify finer granularity options and prompt (aka <em>rubrik section</em>). See example of more fine-grained configuration at `config/evaluator/default_multi_qa.yaml`. 

```bash
python3 eval.py --experiments_folder experiments/ --llm_batch_size 16 --split 'dev' --vllm  --llm_prompt default_multi_qa
```


If you have local ollama server running, you can call models installed on this server as following:

```bash
python3 eval.py --experiments_folder experiments/ --llm_ollama "phi3:latest" --ollama_url "http://localhost:11434"   --llm_prompt default_multi_qa
```




## Training
For training a model add a training config e.g. `train=lora` as an argument, e.g.
```bash
  python3 bergen.py retriever="bm25" reranker="minilm6" generator='tinyllama-chat' dataset='kilt_nq' train='lora'
```

For training the `dev` dataset split that is defined in the config is split in `train` and `test` splits ( default test size: `0.01`). The best model (according to the newly generated `test` split) is loaded after the training and evaluated on the `dev`  dataset split.

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

## Multilingual experiments
See here our [multilingual RAG guide](documentations/multilingual.md) to run experiments with multilingual user queries and/or multilingual Wikipedia as a datastore.

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
## Cite 

If you use BERGEn for your research please consider citing us: 

```
@misc{rau2024bergenbenchmarkinglibraryretrievalaugmented,
      title={BERGEN: A Benchmarking Library for Retrieval-Augmented Generation}, 
      author={David Rau and Hervé Déjean and Nadezhda Chirkova and Thibault Formal and
      Shuai Wang and Vassilina Nikoulina and Stéphane Clinchant},
      year={2024},
      eprint={2407.01102},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01102}, 
}
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
