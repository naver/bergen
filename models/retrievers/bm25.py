'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from pyserini.search.lucene import LuceneSearcher
import subprocess
from tqdm import tqdm
import os
import json


class BM25:

    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self, dataset, index_path=None,  top_k_documents=1, prebuild_index=None, batch_size=256, num_threads=1, return_docs=False):
        self.num_threads = num_threads
        if prebuild_index:
            print(f'Loading Pre-build Index: {prebuild_index}')
            model = LuceneSearcher.from_prebuilt_index(prebuild_index)
        else:
            model = LuceneSearcher(index_path)
        # Batch search
        q_ids, doc_ids, scores, docs = list(), list(), list(), list()
        for examples in tqdm(dataset.iter(batch_size=batch_size), desc=f'Retrieving with {num_threads} threads.'):
            q_id_batch, query_batch = examples['id'], examples['content']
            all_hits = model.batch_search(queries=query_batch, qids=q_id_batch, k=top_k_documents, threads=self.num_threads)
            for q_id, hits in all_hits.items():
                q_scores, q_doc_ids, q_docs = list(), list(), list()
                for hit in hits:
                    doc_id = hit.docid
                    score = hit.score
                    if return_docs:
                        doc = json.loads(model.doc(doc_id).get("raw"))['contents']
                        q_docs.append(doc)
                    q_doc_ids.append(doc_id)
                    q_scores.append(score)
                if len(hits) > 0:
                    doc_ids.append(q_doc_ids)
                    scores.append(q_scores)
                    q_ids.append(q_id)
                    if return_docs:
                        docs.append(q_docs)           
        return {
            "q_id": q_ids if len(doc_ids) > 0 else None,
            "doc_id": doc_ids if len(doc_ids) > 0 else None, 
            "score": scores if len(doc_ids) > 0 else None,
            "doc": docs if return_docs else None
        }

    def index(self, dataset, dataset_path, num_threads=1):
        print('bm25 indexing called')
        if not os.path.exists(dataset_path):
            json_folder = f'{dataset_path}_json/'
            self.save_documents_to_json(dataset, json_folder) 
            os.makedirs(dataset_path)
            self.run_index_process(dataset_path, json_folder, num_threads)
            #print('Removing tmp files.')
            #shutil.rmtree(json_folder)
        return 

    def run_index_process(self, out_folder, json_folder, num_threads):
        command = [
            'python3', '-m', 'pyserini.index.lucene',
            '--collection', 'JsonCollection',
            '--input', json_folder,
            '--index', out_folder,
            '--generator', 'DefaultLuceneDocumentGenerator',
            '--threads', str(num_threads)
            # '--storePositions', '--storeDocvectors', '--storeRaw'
        ]
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")


    def save_documents_to_json(self, dataset, output_folder, max_per_file=250000):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            # Save each chunk to a separate JSON file
            for i in tqdm(range(0, len(dataset), max_per_file), desc='Saving dataset to json.'):
                chunk = dataset[i:i + max_per_file]
                chunk_id = chunk['id']
                chunk_sent = chunk['content']
                formatted_chunk = [{"id": id_, "contents": sent} for id_ , sent in zip(chunk_id, chunk_sent)]
                output_file_path = os.path.join(output_folder, f"doc{i}.json")
                with open(output_file_path, "w") as output_file:
                    json.dump(formatted_chunk, output_file, indent=2)
        else:
            print(f'Skipping writing dataset to json because {output_folder} already exists.')
