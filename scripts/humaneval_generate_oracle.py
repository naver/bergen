import datasets
from tqdm import tqdm
import sys
sys.path.append("../")
from utils import get_oracle_ranking_filename

run_folder = '../runs'
split = 'dev'

humaneval_dataset = datasets.load_from_disk("../datasets/CodeRAGBench_HumanEval_train")
coderagbench_database = datasets.load_from_disk("../datasets/CodeRAGBench_programming_solutions_train")
out_file = get_oracle_ranking_filename(run_folder, "CodeRAGBench_HumanEval", split)
print("Writing to", out_file)
print(humaneval_dataset)
print(coderagbench_database)
with open(out_file, 'w') as fout:
    for index,sample in enumerate(tqdm(humaneval_dataset)):
        for i,sample_db in enumerate(coderagbench_database):
            # look for function signature in database content
            if "def "+sample['entry_point']+"(" in sample_db['content']:
                query_id = str(sample['id'])
                passage_id = str(sample_db['id'])
                fout.write(f'{query_id}\tq0\t{passage_id}\t0\t1\trun\n')
                break
print("Done.")