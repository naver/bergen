"""
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license

A retrieval module for ranking models.
Example: bm25, splade, oracle_provenance. See models/retrievers/ for specific retrievers.
"""

# Retrieve
from tqdm import tqdm
import sys
from torch.utils.data import DataLoader
import torch
import os
import glob
from hydra.utils import instantiate
from utils import load_embeddings


class Retrieve:
    def __init__(
        self,
        init_args=None,
        batch_size=128,
        batch_size_sim=1024,
        pyserini_num_threads=1,
        continue_batch=None,
    ):

        self.continue_batch = continue_batch
        self.batch_size = batch_size
        self.batch_size_sim = batch_size_sim
        self.pyserini_num_threads = pyserini_num_threads
        # instaniate model
        self.model = instantiate(init_args)

    def index(self, dataset, index_path, query_or_doc, overwrite_index=False):
        dataset = dataset[query_or_doc]
        # if dataset has not been encoded before
        if (
            not os.path.exists(index_path)
            or self.continue_batch != None
            or overwrite_index
        ):
            if self.model.model_name == "bm25" and query_or_doc == "doc":
                self.model.index(
                    dataset, index_path, num_threads=self.pyserini_num_threads
                )
            elif self.model.model_name == "oracle_provenance":
                pass
            elif self.model.model_name == "bm25" and query_or_doc == "query":
                pass
            else:
                dataset = dataset.remove_columns(["id"])
                _ = self.encode_and_save(
                    dataset, save_path=index_path, query_or_doc=query_or_doc
                )
                # self.save_index(embs, index_path)

    def retrieve(
        self,
        dataset,
        query_embeds_path,
        doc_embeds_path,
        top_k_documents,
        return_docs=False,
        overwrite_index=False,
    ):

        # index if index doesn't exist
        self.index(
            dataset,
            query_embeds_path,
            query_or_doc="query",
            overwrite_index=overwrite_index,
        )
        self.index(
            dataset,
            doc_embeds_path,
            query_or_doc="doc",
            overwrite_index=overwrite_index,
        )

        doc_ids = dataset["doc"]["id"]
        q_ids = dataset["query"]["id"]

        if self.model.model_name == "bm25":
            bm25_out = self.model(
                dataset["query"],
                index_path=doc_embeds_path,
                top_k_documents=top_k_documents,
                batch_size=self.batch_size,
                num_threads=self.pyserini_num_threads,
                return_docs=return_docs,
            )

            return bm25_out
        else:

            query_embeds = load_embeddings(query_embeds_path)
            query_embeds = query_embeds.to_dense().to("cuda")
            # query_embeds = query_embeds.to('cuda')
            self.model.model = self.model.model.to("cpu")

            # separate query embedding in chunks
            chunks = torch.split(query_embeds, self.batch_size_sim, dim=0)
            scores_sorted_topk, indices_sorted_topk, embeds_sorted_top_k = (
                list(),
                list(),
                list(),
            )

            emb_files = glob.glob(f"{doc_embeds_path}/*.pt")
            sorted_emb_files = sorted(
                emb_files, key=lambda x: int("".join(filter(str.isdigit, x)))
            )

            doc_embeds = list()
            for emb_file in tqdm(
                sorted_emb_files,
                total=len(sorted_emb_files),
                desc=f"Load embeddings and retrieve...",
            ):
                emb_chunk = torch.load(emb_file)
                doc_embeds.append(emb_chunk)

            for chunk in tqdm(chunks, desc=f"Retrieving docs...", total=len(chunks)):
                (
                    scores_sorted_topk_chunk,
                    indices_sorted_topk_chunk,
                    embeds_sorted_top_k_chunk,
                ) = self.load_collection_and_retrieve(
                    chunk,
                    doc_embeds,
                    top_k_documents,
                    dataset_size=len(dataset["doc"]),
                    return_embeddings=False,
                )
                scores_sorted_topk.append(scores_sorted_topk_chunk)
                indices_sorted_topk.append(indices_sorted_topk_chunk)

            scores_sorted_topk = torch.cat(scores_sorted_topk, dim=0)
            indices_sorted_topk = torch.cat(indices_sorted_topk, dim=0)

            # Use sorted top-k indices indices to retrieve corresponding document IDs
            doc_ids = [[doc_ids[i] for i in q_idxs] for q_idxs in indices_sorted_topk]
            return {"score": scores_sorted_topk, "q_id": q_ids, "doc_id": doc_ids}

    @torch.no_grad()
    def encode_and_save(self, dataset, save_path, query_or_doc, chunk_size=150000):
        save_every_n_batches = chunk_size // self.batch_size
        total_n_batches = len(dataset) // self.batch_size + int(
            bool(len(dataset) % self.batch_size)
        )
        # make index folder if save_path is provided
        os.makedirs(save_path, exist_ok=True)
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            # collate_fn=lambda batch: self.model.collate_fn(batch, query_or_doc) if query_or_doc != None else self.model.collate_fn(batch),
            collate_fn=lambda batch: self.model.collate_fn(batch, query_or_doc),
            num_workers=4,
        )
        embs_list = list()
        self.model.model = self.model.model.to("cuda")
        for i, batch in tqdm(
            enumerate(dataloader),
            total=total_n_batches,
            desc=f"Encoding: {self.model.model_name}",
            file=sys.stderr,
        ):
            if self.continue_batch != None:
                if i <= self.continue_batch:
                    continue
            outputs = self.model(query_or_doc, batch)
            emb = outputs["embedding"]
            if save_path != None:
                emb = emb.detach().cpu()
            embs_list.append(emb)
            # save chunk if save_path provided
            if i % save_every_n_batches == 0 and i != 0 or i == total_n_batches - 1:
                chunk_save_path = self.get_chunk_path(save_path, i)
                embs = torch.cat(embs_list)
                if "splade" in self.model.model_name:
                    embs = embs.to_sparse()
                torch.save(embs, chunk_save_path)
                embs_list = list()
        self.model.model = self.model.model.to("cpu")

        return None

    @torch.no_grad()
    def load_collection_and_retrieve(
        self,
        emb_q,
        doc_embeds,
        top_k_documents,
        detach_and_cpu=True,
        return_embeddings=False,
        dataset_size=None,
    ):

        top_k_scores_list, top_k_indices_list, top_k_embed_list = [], [], []

        num_emb = 0
        for emb_chunk in doc_embeds:
            emb_chunk = emb_chunk.to("cuda")
            scores_q = self.model.similarity_fn(emb_q, emb_chunk)
            # if detach_and_cpu:
            #     scores_q = scores_q.detach().cpu().float()
            scores_sorted_q, indices_sorted_q = torch.topk(
                scores_q, top_k_documents, dim=1
            )
            top_k_scores_list.append(scores_sorted_q)
            top_k_indices_list.append(indices_sorted_q + num_emb)
            if return_embeddings:
                emb_chunk_dense = emb_chunk.to_dense()
                top_k_embeddings = emb_chunk_dense[indices_sorted_q]
                top_k_embed_list.append(top_k_embeddings)
            num_emb += emb_chunk.shape[0]
        if num_emb != dataset_size:
            raise IOError(
                f"!!! Index is not complete. Please re-index. Missing {dataset_size-num_emb} documents in the index. !!!"
            )

        # Concatenate top-k scores and indices from each chunk
        all_top_k_scores = torch.cat(top_k_scores_list, dim=1).detach().cpu()
        all_top_k_indices = torch.cat(top_k_indices_list, dim=1).detach().cpu()

        if return_embeddings:
            all_top_k_embeds = torch.cat(top_k_embed_list, dim=1).to("cpu")
        # Get final top-k scores and indices across all chunks
        final_top_k_scores, top_k_indices = torch.topk(
            all_top_k_scores.float(), top_k_documents, dim=1
        )
        # Extract corresponding indices for final top-k scores
        final_top_k_indices = torch.gather(all_top_k_indices, 1, top_k_indices)

        if return_embeddings:
            final_top_k_embeddings = torch.gather(
                all_top_k_embeds,
                1,
                top_k_indices.unsqueeze(-1).expand(-1, -1, all_top_k_embeds.shape[-1]),
            )

            return final_top_k_scores, final_top_k_indices, final_top_k_embeddings
        else:
            return final_top_k_scores, final_top_k_indices, None

    def tokenize(self, example):
        return self.model.tokenize(example)

    def get_clean_model_name(self):
        return self.model.model_name.replace("/", "_")

    def get_chunk_path(self, save_path, chunk):
        return f"{save_path}/embedding_chunk_{chunk}.pt"
