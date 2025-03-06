# Provence

This page contains code and experimental data for [Provence: efficient and robust context pruning for retrieval-augmented generation](https://openreview.net/forum?id=TDy5Ih78b4) (ICLR'25)

[[HF model](https://huggingface.co/naver/provence-reranker-debertav3-v1)] [[blogpost](https://huggingface.co/blog/nadiinchi/provence)] [[Paper](https://openreview.net/forum?id=TDy5Ih78b4&noteId=TDy5Ih78b4)] [[arxiv](https://arxiv.org/abs/2501.16214)]

The code is currently in [pull-request](https://github.com/naver/bergen/pull/40) and will be merged soon.

_Provence_ is a method for training a lightweight __context pruning model__ for retrieval-augmented generation, particularly optimized for question answering. Given a user question and a retrieved passage, Provence __removes sentences from the passage that are not relevant to the user question__. This __speeds up generation__ and __reduces context noise__, in a plug-and-play manner __for any LLM or retriever__. 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/63bea3d44a2beec6555fd7dc/N1luvOjp7EJ-I-EcLFgV6.png)

### Our experimental data

Zip archive with our experimental data is available at this [Google drive link](https://drive.google.com/file/d/1hjcoCUniGS5bjF7xZYzi8s1HXJOdJwWL/view?usp=sharing)

Each subfolder contains:
*  `eval_dev_metrics.json`: reports various metrics (exact match, match, recall, LLMEval etc) for each run
*  `eval_dev_out.json`: contains generations for each run (from `llama-2-7b-chat`)
*  `.trec` runs: only available for `baseline` methods (==full context without context compression), since retrieval+reranking result is the same across all context compression methods.

After downloading and unzipping this arxhive, you can use `Plot.ipynb` to plot the main Figure 2 from the paper.

### Running evaluation

[readme in progress]

### Training models

Code for model training is provided in the folder where this readme is located.

Example command:
```
python3 train_provence.py --exp_folder $EXP_FOLDER \
                          --model $MODEL 
                          --lr $LR \ 
                          --train_batch_size $BS \ 
                          --epochs $EPOCHS \
                          --training_type $TRAINING_TYPE \ # compression only or joint training
                          --data_path path_to_nq_data path_to_msmarco_data \
                          --run_path path_to_training_run_nq path_to_training_run_msmarco \  # contains all the top_k documents for each query in the train set
                          --ranking_validation_data rerank_file.trecdl19.spladev3.top50.tsv \
                          --ranking_validation_qrels qrel_trecdl19.json
```

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
