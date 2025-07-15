# Provence

This page contains code and experimental data for [Provence: efficient and robust context pruning for retrieval-augmented generation](https://openreview.net/forum?id=TDy5Ih78b4) (ICLR'25)

[HF model](https://huggingface.co/naver/provence-reranker-debertav3-v1) | [blogpost](https://huggingface.co/blog/nadiinchi/provence) | [paper](https://openreview.net/forum?id=TDy5Ih78b4&noteId=TDy5Ih78b4) | [arxiv](https://arxiv.org/abs/2501.16214)

_Provence_ is a method for training a lightweight __context pruning model__ for retrieval-augmented generation, particularly optimized for question answering. Given a user question and a retrieved passage, Provence __removes sentences from the passage that are not relevant to the user question__. This __speeds up generation__ and __reduces context noise__, in a plug-and-play manner __for any LLM or retriever__. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63bea3d44a2beec6555fd7dc/N1luvOjp7EJ-I-EcLFgV6.png)

### Citation
```
@inproceedings{
chirkova2025provence,
title={Provence: efficient and robust context pruning for retrieval-augmented generation},
author={Nadezhda Chirkova and Thibault Formal and Vassilina Nikoulina and St{\'e}phane Clinchant},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=TDy5Ih78b4}
}
```


### Our experimental data

Zip archive with our experimental data is available at this [Google drive link](https://drive.google.com/file/d/1hjcoCUniGS5bjF7xZYzi8s1HXJOdJwWL/view?usp=sharing)

Each subfolder contains:
*  `eval_dev_metrics.json`: reports various metrics (exact match, match, recall, LLMEval etc) for each run
*  `eval_dev_out.json`: contains generations for each run (from `llama-2-7b-chat`)
*  `.trec` runs: only available for `baseline` methods (==full context without context compression), since retrieval+reranking result is the same across all context compression methods.

After downloading and unzipping this archive, you can use [Provence_main_plot.ipynb](https://github.com/naver/bergen/blob/main/scripts/provence/Provence_main_plot.ipynb) to plot the main Figure 2 from the paper.

### Installation
This page is a part of the [Bergen](https://arxiv.org/abs/2407.01102) library (BEnchmarking for Retrieval-augmented GEneration), please follow the [main installation guidlines](https://github.com/naver/bergen/blob/main/documentation/INSTALL.md).

Depending on the particular context compressors that you plan to use, you may also need the following libraries:
* `nltk` (for Provence and Recomp)
```
pip install nltk
python -c "import nltk; nltk.download('punkt_tab')"
```
* `llmlingua` (for LLMLingua family)
```
pip install llmlingua
```

### Running evaluation

Script [run_exps.sh](https://github.com/naver/bergen/blob/main/scripts/provence/run_exps.sh) lists all configurations needed to compute the main Figure 2 in the paper. Each run in the script  launches [bergen.py evaluation](https://github.com/naver/bergen/tree/provence?tab=readme-ov-file#quick-start) of the Bergen library.

We compare the following context pruners for RAG:
* Provence (ours): https://arxiv.org/abs/2501.16214 (standalone and a unified reranking+context pruning model)
* RECOMP: https://arxiv.org/abs/2310.04408 (extractive and abstractive)
* LLMLingua2 and LongLLMLingua: https://llmlingua.com/llmlingua2.html
* DSLR: https://arxiv.org/abs/2407.03627 (concurrent work)

In Bergen, the usage of a context pruner is enabled as follows:
```
python3 bergen.py +context_processor=PROCESSOR ...
```
Replace `PROCESSOR` with a name of a particular config from [config/context_processor](https://github.com/naver/bergen/tree/main/config/context_processor), code for all context processors is located in [models/context processors](https://github.com/naver/bergen/tree/main/models/context_processors). See script [run_exps.sh](https://github.com/naver/bergen/blob/main/scripts/provence/run_exps.sh) which enumerates all the settings we used.

Examples:

```
python3 bergen.py +context_processor=provence/provence_rerank_0.1 dataset=multidomain/pubmed_bioasq11b_ragged retriever='splade-v3' ++generation_top_k=50 generator='vllm_llama-2-7b-chat'
```

```
python3 bergen.py +context_processor=provence/provence_standalone_0.1 dataset=multidomain/pubmed_bioasq11b_ragged retriever='splade-v3' reranker='debertav3' ++generation_top_k=5 generator='vllm_llama-2-7b-chat'
```

### Training Provence models

Training consists of three steps: (1) retrieval+reranking; (2) data labeling; (3) training Provence model. We also provide below the [data](https://github.com/naver/bergen/blob/main/scripts/provence/readme.md#our-training-data-obtained-after-steps-1--2) obtained after steps 1 and 2 so that you can jump directly to [step 3](https://github.com/naver/bergen/tree/main/scripts/provence#step-3-training-provence).

If you have any questions, do not hesitate to contact us by corresponding emails specified in the [paper](https://openreview.net/forum?id=TDy5Ih78b4&noteId=TDy5Ih78b4)!

#### Step 1: retrieval+reranking

The purpose of this step is to save Bergen datasets for training _queries_ and _datastore_, and also to compute and save Bergen _.trec runs_, i.e. the results of retrieval and reranking for training queries:

```
cd ../..
python3 bergen.py dataset=msmarco-docs-chunked +dataset_split=train retriever="splade-v3" reranker="debertav3"
```

Warning: indexing, retrieval and reranking can take a long time, i.e. days... :(
Do not forget your CUDA / sbatch settings!

#### Step 2: oracle labeling

To train Provence, we label data using LLama-3-b8 (we used the 3.1 version, we did not try newer versions / model):

```
python3 gen_silver_labeling_provence.py --queries "../../datasets/ms-marco-docs-v1-queries-dev_full" --datastore "../../datasets/ms-marco-docs-v1-chunked-v1_full" --trec "../../runs/run.rerank.retriever.top_50.naver_splade-v3.rerank.top_50.ms-marco-docs-v1-queries-dev.ms-marco-docs-v1-chunked-v1.dev.naver_trecdl22-crossencoder-debertav3.trec" --outdir PATH_TO_DATA
```

Specify your `PATH_TO_DATA` where to save the results of the labeling. The script will create a separate `.json` file per a query-context pair in this `PATH_TO_DATA` folder (total size several Gb). Other files specified in arguments should have been saved by Bergen in step 1.

Warning: this step can also take 2-3 days on a Tesla A100 GPU. Do not forget your CUDA / sbatch settings!

The script for labeling `gen_silver_labeling_provence.py` is provided in the same folder as this readme.

#### Step 3: training Provence

Example command:
```
python3 train_provence.py --exp_folder $EXP_FOLDER \ # where to save the checkpoints / logs
                          --model $MODEL # default: "naver/trecdl22-crossencoder-debertav3"
                          --lr $LR \ # default: 3e-06
                          --train_batch_size $BS \ # default: 48
                          --epochs $EPOCHS \ # default: 1
                          --training_type $TRAINING_TYPE \ # "joint" or "compression"
                          --data_path $PATH_TO_DATA \
                          --run_path $PATH_TO_RUN \  # contains all the top_k documents for each query in the train set
                          --ranking_validation_data data/rerank_file.trecdl19.spladev3.top50.tsv \
                          --ranking_validation_qrels data/qrel_trecdl19.json
```

The script for training `train_provence.py` is provided in the same folder as this readme.

* `PATH_TO_DATA`: the data folder generated in step 2
* `TRAINING_TYPE`: "joint" for joint reranking+context compression model, "compression" for compression-only (standlone) model, "ranking" for reranking only
* `PATH_TO_RUN`: `.trec` file generated in step 1 (only needed to train the reranking head).
* `EXP_FOLDER`: your custom folder where to save the checkpoints and the logs

Do not forget your CUDA / sbatch settings! One epoch on the full MS Marco data takes several days, but you can get good results with smaller data, e.g. 1/10 of the whole set.


### Our training data obtained after steps 1 & 2

Here you can find zip archives for MS Marco and NQ data, for context pruning:

MS Marco: https://drive.google.com/file/d/1UYyZlB-t_T3uloPb5dJxCVWn22IQOP6s/view?usp=sharing

NQ: https://drive.google.com/file/d/18qBPAoAmJGXrdVzpOHgylTOyCEIl_hk5/view?usp=sharing

Each archive includes a set of jsons, each json contains a question-context pair & a LLama-3-8B-generated answer & extracted indexes of relevant sentences.

Here are the corresponding trec runs:

MS Marco: https://drive.google.com/file/d/1A_IZMDYxzZHjmG4lCfNySI3wGCMd6IWG/view?usp=sharing

NQ: https://drive.google.com/file/d/1ZVx7rUygGE1qZfDdaly2xoSEZ6h560ma/view?usp=sharing

We train the final model on a mix of both, but training separately on one of them also gives good results. MS Marco is larger and leads to a bit better model, NQ is smaller -> faster training.
