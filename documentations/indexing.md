
Datasets will be downloaded, pre-processed, indexed and saved if they do not exist yet, otherwise they will be loaded from `dataset_folder` and `index_folder` respectively. 



#Existing Runs
 - Download from NLE servers
 - Example using an existing run
-  

```

CONFIG=rag python3 main.py  retriever=spladev3 reranker=debertav3 dataset='kilt_nq'
```

All datasets can be overwritten by adding `+overwrite_datasets=True` as an argument (`Caution`: This might overwrite collections that take long long to encode). In case the indexing is interrupted you can continue encoding a collection from batch 1000 by additionally using the argument `+continue_batch=1000`.

