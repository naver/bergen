# XProvence

This page contains code and experimental data for [Zero-Cost Multilingual Context Pruning for Retrieval-augmented Generation
]() (ECIR'26)

[HF model](https://huggingface.co/naver/xprovence-reranker-bgem3-v2) | [paper]() 

_XProvence_ is a multilingual adaptation of Provence supporting 100+ languages. It is a lightweight __context pruning model__ for retrieval-augmented generation, particularly optimized for question answering. Given a user question and a retrieved passage, XProvence __removes sentences from the passage that are not relevant to the user question__. This __speeds up generation__ and __reduces context noise__, in a plug-and-play manner __for any LLM or retriever__. 

![image/png]()

### Citation
```
@inproceedings{

}
```


### Our experimental data

Zip archive with our experimental data is available at this [Google drive link](https://drive.google.com/file/d/1MT6WUqFLpPFPdE1_fSMr33KAi0VSMTTi/)

Each subfolder contains:
*  `eval_dev_metrics.json`: reports various metrics (exact match, match, recall, LLMEval etc) for each run
*  `eval_dev_out.json`: contains generations for each run (from `aya-expanse-8b`)

After downloading and unzipping this archive, you can use [make_pareto_front.ipynb](https://github.com/naver/bergen/blob/main/scripts/xprovence/make_pareto_front.ipynb) to plot the pareto front for one dataset similar to the main figure in the paper. The notebook plots the MKQA plots, in order to plot another benchmark, change the notebook accordingly.

### Installation
This page is a part of the [Bergen](https://arxiv.org/abs/2407.01102) library (BEnchmarking for Retrieval-augmented GEneration), please follow the [main installation guidlines](https://github.com/naver/bergen/blob/main/documentation/INSTALL.md).

Depending on the particular context compressors that you plan to use, you may also need the following libraries:
* `spacy` (for XProvence)
```
pip install spacy
python -m spacy download xx_sent_ud_sm
```

### Running evaluation

We compare the following context pruners for RAG:
* DSLR: https://arxiv.org/abs/2407.03627 

In Bergen, the usage of a context pruner is enabled as follows:
```
python3 bergen.py +context_processor=PROCESSOR ...
```
Replace `PROCESSOR` with a name of a particular config from [config/context_processor](https://github.com/naver/bergen/tree/main/config/context_processor), code for all context processors is located in [models/context processors](https://github.com/naver/bergen/tree/main/models/context_processors).

Examples:

```
python -u bergen.py generator='aya-expanse-8b' retriever='bge-m3' reranker='bge-m3' dataset='mkqa/mkqa_ar.retrieve_ar' prompt='basic_translated_langspec/ar'  +context_processor='provence/xprovence_rerank_0.5'
```

```
python -u bergen.py generator=aya-expanse-8b retriever=oracle_provenance  +context_processor='provence/xprovence_standalone_0.5'   dataset=medexpqaexp/medexpqaexp_it.retrieve_it prompt=medexpqa/it
```

### Training XProvence models

Training consists of three steps: (1) Translating Provence data; (2) Relevance scores calculation; (3) training XProvence model. We also provide below the [translated data](https://drive.google.com/file/d/1i1Qr_ogdS1R7G4At2iIfAuXOmmjlaIMA/) obtained after steps 1 and the [relevance scores](https://drive.google.com/file/d/1XHtN-xVx5kgH5b-O1bQc7FbxB1K3Fgu2/) obtained from step 2 so that you can jump directly to step 3.

If you have any questions, do not hesitate to contact us by corresponding emails specified in the [paper]()!

#### Step 1: Data Translation

The purpose of this step is to translate the English MSMARCO data used to train Provence. The English MSMARCO data can be found [here](https://drive.google.com/file/d/1UYyZlB-t_T3uloPb5dJxCVWn22IQOP6s/). In XProvence, we sampled 125K examples for each language which are then translated to the respective language. The script to translate into any language can be found [here](https://github.com/naver/bergen/blob/main/scripts/xprovence/translation/translate.py). To run the script for a single language, 
```
python -u translate.py --input_dir <PATH_TO_125K_SAMPLES> --out_dir <OUTPUT_DIRECTORY> --lang <LANGUAGE>
```

#### Step 2: Relevance Scores

We utilized the [BGE-m3 reranker model](https://huggingface.co/BAAI/bge-reranker-v2-m3) to calculate the relevance scores after translation since it can differ from the English scores. We follow the same instructions from the [HF page](https://huggingface.co/BAAI/bge-reranker-v2-m3) to compute the scores.

#### Step 3: training XProvence

Example command:
```
torchrun --nproc_per_node=4 \
         --nnodes=1 \
         --node_rank=0 \
         --master_addr=localhost \
         --master_port=12345 \
         xprovence_training.py \
            --distributed \
            --exp_folder $EXP_FOLDER \
            --model $PATH_TO_MODEL \
            --tokenizer_name $PATH_TO_MODEL \
            --lr 0.00001  \
            --train_batch_size 32 \
            --eval_batch_size 32 \
            --epochs 3 \
            --training_type $TRAINING_TYPE \
            --post_process \
            --data_path $PATH_TO_TRANSLATED_DATA \
            --run_path $PATH_TO_RUN \
            --ranking_validation_data data/rerank_file.trecdl19.spladev3.top50.tsv \
            --ranking_validation_qrels data/qrel_trecdl19.json \
```

The script for training `train_xprovence.py` is provided in the same folder as this readme.

* `EXP_FOLDER`: your custom folder where to save the checkpoints and the logs
* `PATH_TO_MODEL`: path to HF model.
* `PATH_TO_TRANSLATED_DATA`: the translated data folder created in step 1
* `TRAINING_TYPE`: "joint" for joint reranking+context compression model, "compression" for compression-only (standlone) model, "ranking" for reranking only
* `PATH_TO_RUN`: `.trec` file generated in step 2 (only needed to train the reranking head).


### Our training data obtained after steps 1 & 2

Here you can find zip archives for MS Marco and NQ data, for context pruning:

English MS Marco: https://drive.google.com/file/d/1UYyZlB-t_T3uloPb5dJxCVWn22IQOP6s/view?usp=sharing

Translated Data: https://drive.google.com/file/d/18qBPAoAmJGXrdVzpOHgylTOyCEIl_hk5/view?usp=sharing

Each archive includes a set of jsons, each json contains a question-context pair & an answer & extracted indexes of relevant sentences.

Here are the corresponding Relevance scores .trec files:

Translated Data: https://drive.google.com/file/d/1XHtN-xVx5kgH5b-O1bQc7FbxB1K3Fgu2/