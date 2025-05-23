### Configs for BERGEN

Usage Examples

1. Generation without Retrieval (Closed Book):
   ```bash
   python3 bergen.py generator='tinyllama-chat' dataset='kilt_nq'
   ```

2. Retriever - only first stage:
   ```bash
   python3 bergen.py retriever="splade-v3" generator='tinyllama-chat' dataset='kilt_nq'
   ```

3. Retriever + Reranker:
   ```bash
   python3 bergen.py retriever="splade-v3" reranker="debertav3" generator='tinyllama-chat' dataset='kilt_nq'
   ```
4. Change the generator without a config file:
   Provide just the name of an existing HuggingFace model as model_name argument.
   ```bash
   python3 bergen.py  retriever="splade-v3" reranker="debertav3" generator="hf" generator.init_args.model_name="meta-llama/Llama-2-7b-hf"   dataset=kilt_nq
   ```
   For additional options such as batch size or quantization, please change the file or create a config

5. Using vllm for faster generation:
   ```bash
   python3 bergen.py retriever="splade-v3" reranker="debertav3" generator='vllm_SOLAR-107B' dataset='kilt_nq'
   ```


All those different pipelines rely on  configuration files located in the config dir.
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

To add a new datasets, please refer to an following guide:[extensions](extensions.md)

### Retrieval
Indexing of the document collections will be automatically launched if needed: retrieval, reranking runs will be loaded from files if they already exist in `runs`, otherwise they will be created.  Retrieval will only be evaluated if the `query` dataset contains the field `ranking_label`.
For details about indexing, please refer to [indexing.md](indexing.md)


Experiments are saved under `experiments_folder`. The experiment folder is named after the hash of the config, unless the experiment is finished the folder name will contain the prefix `tmp_`. You can provide a custom name for the experiment folder by adding `+run_name={YOUR_NAME}`. The script will be aborted if an experiment with the exact same parameters has been run before. To overwrite the experiment add `+overwrite_exp=True` as an argument.


- To overwrite the experiment add `+overwrite_exp=True` as an argument, due to a bug or another update in the config 
-  To rebuild the index (and subsequently the ranking run) add `+overwrite_index=True` as an argument.

To print the results in a table run the following commands. By default, this will print all experiments that contain generation metric files in `experiments/` and sort them by the `generator`.

```bash
# will print a markdown of the results and save a csv file under the results directory
python3 print_results.py --csv --folder experiments/

#csv files with all the metrics
exp_folder,Retriever,P_1,Reranker,Generator,gen_time,query_dataset,r_top,rr_top,M,EM,F1,P,R,Rg-1,Rg-2,Rg-L,BEM,LLMeval
216567b3d48ef3fc,naver/splade-v3/,,naver/trecdl22-crossencoder-debertav3,TinyLlama/TinyLlama-1.1B-Chat-v1.0,00:03:53.19,KILTTriviaqa,100,100,0.6763772175536882,0.00018674136321195143,0.11749967712256401,0.07122756370055569,0.5380933823321367,0.1505780809175042,0.055962386132169924,0.14611799602749245,0.47356051206588745,
```


### Generator
A typical generator configuration:
```
init_args: 
  _target_: models.generators.llm.LLM
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  max_new_tokens: 128
  max_length: 32000
  batch_size: 256
  max_doc_len: 100
  use_middle_truncation: True
```
Hydra instantiates the given `_target_` class passing the rest of the arguments to the class constructor. Here:
- max_length: the maximum length (in tokens) of the full prompt passed to the generator
- max_new_tokens: max number of generated tokens
- max_doc_len: each retrieved document is cropped to max_doc_len words before being appended to the query
- use_middle_truncation: when the input prompt exceeds max_length, the middle part of the prompt is removed