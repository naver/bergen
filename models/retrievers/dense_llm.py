"""
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
"""

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel, PeftConfig
from models.retrievers.retriever import Retriever


class DenseLLM(Retriever):
    def __init__(
        self,
        model_name,
        similarity,
        max_len=512,
        adapters=True,
        prompt_q=None,
        prompt_d=None,
    ):
        self.model_name = model_name
        self.max_len = max_len
        self.prompt_q = "" if prompt_q is None else prompt_q
        self.prompt_d = "" if prompt_d is None else prompt_d
        self.similarity = similarity
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the tokenizer and model
        if "repllama" in model_name:
            # repllama does not have a proper config to load the tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf", padding_side="right"
            )
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, padding_side="right"
            )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = self.get_model(model_name, adapters)
        self.tokenizer.model_max_length = self.max_len

    def get_model(self, model_name, adapters):
        if adapters:
            config = PeftConfig.from_pretrained(model_name)
            base_model = AutoModel.from_pretrained(
                config.base_model_name_or_path,
                device_map="auto",
                torch_dtype="auto",
            )
            model = PeftModel.from_pretrained(base_model, model_name)
            model = model.merge_and_unload()
        else:
            # not sure about the BitsAndBytesConfig thing;
            # bnb_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_compute_dtype=torch.float16,
            # )
            # model = AutoModel.from_pretrained(
            #     model_name,
            #     device_map="auto",
            #     torch_dtype=torch.bfloat16,
            #     quantization_config=bnb_config,
            # )
            model = AutoModel.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=True,
            )
        model.eval()
        return model

    def collate_fn(self, batch, query_or_doc=None):
        key = "generated_query" if query_or_doc == "query" else "content"
        content = [sample[key] for sample in batch]
        if query_or_doc == "doc":
            content = [f"{self.prompt_d}{text}" for text in content]
        else:
            content = [f"{self.prompt_q}{text}" for text in content]
        # slighlty adapted from: https://huggingface.co/samaya-ai/promptriever-mistral-v0.1-7b-v1
        batch_dict = self.tokenizer(
            content,
            max_length=self.max_len - 1,
            return_token_type_ids=False,
            return_attention_mask=False,
            padding=False,
            truncation=True,
        )
        batch_dict["input_ids"] = [
            input_ids + [self.tokenizer.eos_token_id]
            for input_ids in batch_dict["input_ids"]
        ]
        return self.tokenizer.pad(
            batch_dict,
            padding=True,
            pad_to_multiple_of=8,
            return_attention_mask=True,
            return_tensors="pt",
        )

    def __call__(self, query_or_doc, kwargs):
        # query_or_doc not used, we assume a single model

        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        # get accumulated eos token counts per example
        accumulated_eos_tokens = (
            kwargs["input_ids"] != self.tokenizer.pad_token_id
        ).cumsum(dim=1)
        # check if accumulated eos token == to the length of example (means that no eos is contained in example because it has been truncated)
        missing_eos_positions = accumulated_eos_tokens == kwargs["input_ids"].size(1)
        kwargs["input_ids"][missing_eos_positions] = self.tokenizer.pad_token_id
        outputs = self.model(**kwargs)
        # pooling over hidden representations
        emb = outputs[0]
        # count of eos tokens per example
        num_eos = (kwargs["input_ids"] == self.tokenizer.pad_token_id).cumsum(dim=1)
        # get index of first occurence of eos
        first_eos_indices = (num_eos == 1).nonzero()[:, 1]
        # gather the embeddings based on the first EOS indices
        first_eos_emb = emb[torch.arange(emb.size(0)), first_eos_indices]
        return {"embedding": first_eos_emb}

    def similarity_fn(self, query_embds, doc_embds):
        return self.similarity.sim(query_embds, doc_embds)


class DotProduct:

    @staticmethod
    def sim(query_embds, doc_embds):
        return torch.mm(query_embds, doc_embds.t())


class CosineSim:

    @staticmethod
    def sim(query_embds, doc_embds):
        query_embds = query_embds / (
            torch.norm(query_embds, dim=-1, keepdim=True) + 1e-9
        )
        doc_embds = doc_embds / (torch.norm(doc_embds, dim=-1, keepdim=True) + 1e-9)
        return torch.mm(query_embds, doc_embds.t())
