'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import PeftModel, PeftConfig
from models.retrievers.retriever import Retriever


class RepLlama(Retriever):
    def __init__(self, model_name=None,max_len=512):

        self.max_len= max_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the tokenizer and model
        if 'repllama' in model_name:
            self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='right')
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right')

        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = self.get_model(model_name)
        self.tokenizer.model_max_length = 512
        self.model_name = model_name
        self.model.config.use_cache = False


    def get_model(self, model_name):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        if 'repllama' in model_name:
            config = PeftConfig.from_pretrained(model_name)
            base_model = AutoModel.from_pretrained(config.base_model_name_or_path, device_map='auto', torch_dtype=torch.float16)
            model = PeftModel.from_pretrained(base_model, model_name)
            model = model.merge_and_unload()
        else:
            model = AutoModel.from_pretrained(model_name, device_map='auto', torch_dtype=torch.bfloat16, quantization_config=bnb_config)
        model.eval()
        return model

    def collate_fn(self, batch, query_or_doc):
        if query_or_doc == 'doc':
            content = [f"query: {sample['content']}{self.tokenizer.eos_token}" for sample in batch]
        else:
            content = [f"passage: {sample['generated_query']}{self.tokenizer.eos_token}" for sample in batch]

        return_dict = self.tokenizer(content, padding=True, truncation=True, max_length=self.max_len,return_tensors='pt')
        return return_dict

    def __call__(self, kwargs):
        kwargs = {key: value.to(self.device) for key, value in kwargs.items()}
        # get accumulated eos token counts per exmaple
        accumulated_eos_tokens = (kwargs['input_ids'] != self.tokenizer.pad_token_id).cumsum(dim=1)
        # check if accumulated eos token == to the length of example (means that no eos is contained in example because it has been truncated).
        missing_eos_positions = accumulated_eos_tokens == kwargs['input_ids'].size(1)
        kwargs['input_ids'][missing_eos_positions] = self.tokenizer.pad_token_id
        outputs = self.model(**kwargs)
        # pooling over hidden representations
        emb = outputs[0]
        
        # count of eos tokens per example
        num_eos = (kwargs['input_ids'] == self.tokenizer.pad_token_id).cumsum(dim=1)
        # get index of first occurence of eos 
        first_eos_indices = ( num_eos == 1).nonzero()[:, 1]

        # Gather the embeddings based on the first EOS indices
        first_eos_emb = emb[torch.arange(emb.size(0)), first_eos_indices]
        normed_emb = torch.nn.functional.normalize(first_eos_emb, p=2, dim=1)

        return {
                "embedding": normed_emb
            }

    def similarity_fn(self, query_embds, doc_embds):
        return torch.mm(query_embds, doc_embds.t())

