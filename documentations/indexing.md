
In generaal, datasets will be downloaded, pre-processed, indexed and saved if they do not exist yet, otherwise they will be loaded from `dataset_folder` and `index_folder` respectively. 



### Reuse exiting Retrieval Results 

The retrieval runs are already included in the folder 'runs'. By default, the config
will reuse any existing runs provided here.

### Reusing the index
We provide a dense retriever index (RetroMAE) and a sparse one (splade-v3) which can be found  https://download.europe.naverlabs.com/bergen/

```
mkdir -p indexes
cd indexes
# Example for a dense retriever
wget https://download.europe.naverlabs.com/bergen/kilt-100w_doc_Shitao_RetroMAE_MSMARCO_distill.tar 
tar -xvf kilt-100w_doc_Shitao_RetroMAE_MSMARCO_distill.tar
```

To rerun the retrieval for an experiment, one need to add
```
overwrite_run: true
```
to the config files. Additionally, the number of retrieved document and the reranking depth can be adjusted as well:

```
overwrite_run: true
retrieve_top_k: 100
rerank_top_k: 100
generation_top_k: 5
```
In this case, the first stage retriever will return 100 documents, which will be reranked further. The LLM generator will use the top 5 documents in its context.




### Adding  a new retriever
In this case, follows in instructions in [extensions](extensions.md) to define a new class of retriever. Then, it is require to launch the indexing process only once for whole the QA collections, since it relies on the same Wikipedida document collections. To do so, one can launch the following command line with **only** the retriever specified as followed

```
CONFIG=rag python3 bergen.py  retriever=mynewretriever dataset='kilt_nq'
```


All datasets can be overwritten by adding `+overwrite_datasets=True` as an argument (`Caution`: This might overwrite collections that take long long to encode). In case the indexing is interrupted you can continue encoding a collection from batch 1000 by additionally using the argument `+continue_batch=1000`.

