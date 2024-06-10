import sys
import os
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_yaml', type=str, required=True, 
                    help="path to the yaml describing the dataset for which indexes will be merged")
parser.add_argument('--indexes_path', type=str, required=True,
                   help="FULL path to indexes folder. It is usually located at the root directory of BERGEN repo, but please be sure to provide the full path.")
parser.add_argument('--retriever', type=str, required=True, 
                    help="name of retriever for which indexes merge will be done. This name is contained in index directory names")
# example to merge wikipedias in all languages: python3 merge_indexes.py --dataset_yaml ../../config/dataset/mkqa/mkqa_ar.retrieve_all.yaml --retriever BAAI_bge-m3 --indexes_path /home/{user}/bergen/indexes/

args = parser.parse_args()

with open(args.dataset_yaml) as stream:
    config = yaml.safe_load(stream)

in_dataset_names = config["dev"]["doc"]["init_args"]["in_dataset_names"]
in_dataset_splits = config["dev"]["doc"]["init_args"]["in_dataset_splits"]
out_dataset_name = config["dev"]["doc"]["init_args"]["out_dataset_name"]
out_dataset_split = config["dev"]["doc"]["init_args"]["split"]
assert len(in_dataset_names) > 1
assert len(in_dataset_names) == len(in_dataset_splits)

out_path = f"{args.indexes_path}/{out_dataset_name}_doc_{args.retriever}"
if os.path.exists(out_path) and len(os.listdir(out_path)) > 0:
    raise f"Folder {out_path} already exists and is not empty, exiting"
os.makedirs(out_path, exist_ok=True)

for in_dataset_name, in_split in zip(in_dataset_names, in_dataset_splits):
    in_path = f"{args.indexes_path}/{in_dataset_name}_doc_{args.retriever}"
    if not os.path.exists(in_path) or len(os.listdir(in_path)) == 0:
        raise f"All indexes for merging should be precomputed. Index {in_path} does not exist"

current_global_index = 0
for in_dataset_name, in_split in zip(in_dataset_names, in_dataset_splits):
    in_path = f"{args.indexes_path}/{in_dataset_name}_doc_{args.retriever}"
    for chunk in sorted(os.listdir(in_path), key=lambda x: int(''.join(filter(str.isdigit, x)))):
        idx = int(''.join(filter(str.isdigit, chunk)))
        newidx = current_global_index + idx
        newchunk = f"embedding_chunk_{newidx}.pt"
        os.system(f"ln -s {in_path}/{chunk} {out_path}/{newchunk}")
    current_global_index = newidx + 1