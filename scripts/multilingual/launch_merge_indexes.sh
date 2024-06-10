INDEX_PATH=/home/nchirkov/gitt/bergen/indexes/

#python3 merge_indexes.py --dataset_yaml ../../config/dataset/mkqa/mkqa_ar.retrieve_all.yaml --retriever BAAI_bge-m3 --indexes_path $INDEX_PATH

for lang in ar zh fi fr de ja it ko pt ru es th; do
    python3 merge_indexes.py --dataset_yaml ../../config/dataset/mkqa/mkqa_${lang}.retrieve_en_${lang}.yaml --retriever BAAI_bge-m3 --indexes_path $INDEX_PATH
done