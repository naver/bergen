import math
import torch
import random
from tqdm import tqdm
from models.generators.generator import Generator
from models.generators.llm import LLM
from torch.utils.data import DataLoader
from utils import prepare_labels


class LLMCocom(Generator):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 checkpoint_path: str,
                 context_max_length: int = 128,
                 max_new_tokens: int = 128,
                 model_max_length: int = 1280,
                 prompt: str = None,
                 compr_rate: float = None):
        """
        Class to use cocom with compression
        checkpoint_path: path to a COCOM checkpoint
        context_max_length: maximum length for encoding documents
        max_new_tokens: maximum number of tokens for generation
        model_max_length: maximum length used in the final query (should be large enough)
        """
        # Lazy import to prevent dependency
        from cocom.model import COCOM, COCOMConfig
        
        Generator.__init__(self, model_name=model_name, batch_size=batch_size)

        # Loading the cocom model:
        if checkpoint_path is not None:
            self.model = COCOM.from_pretrained(checkpoint_path)
        else:
            cfg = COCOMConfig(
                decoder_model_name='mistralai/Mistral-7B-Instruct-v0.2',
                max_new_tokens=128,
                quantization='no',
                compr_model_name=None,
                compr_rate=compr_rate,
                lora=True,
                training_form='both_separately',
                lora_r=16,
                kbtc_training=False,
                optimize_mem_tokens=False
            )
            print('Creating brand new COCOM model:', cfg)
            self.model = COCOM(cfg)
            
        self.model.eval()
        
        self.prompt = prompt
        self.context_max_length = context_max_length
        self.max_new_tokens = max_new_tokens
        self.model_max_length = model_max_length
        
        self.model.bfloat16()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.response_token_ids = self.get_response_template_ids()
                
    def generate(self, instr_tokenized):
        """
        Nothing to do here, just convey to cocom since instr_tokenized went throught the collate_fn
        """
        instr_tokenized = {k: v.to(self.device) for k,v in instr_tokenized.items() if isinstance(v, torch.Tensor)}
        return self.model.generate(instr_tokenized, max_new_tokens=self.max_new_tokens)

    def eval(self, dataset):
        """
        We re-implement this method since we do not want to use the already existing "TokenizedDataset", we need
        to do the tokenization deeper in the process.
        dataset: returned by utils.prepare_dataset_from_ids
        """
        assert len(dataset) > 0, 'Empty dataset'

        # We get here the generation top k value, used in the encoder/decoder routine to batch encodings.
        # ideally we would convey it somewhere else, but we use asserts later on to continuously check it's unchanged
        example_docs = dataset[0]['doc']
        self.model.generation_top_k = len(example_docs)

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                    collate_fn=lambda l: self.collate_fn(l, eval=True), num_workers=4)

        responses, instructions, query_ids, queries, labels, ranking_labels = list(), list(), list(), list(), list(), list()
        with torch.no_grad():
            for data_dict in tqdm(dataloader, desc='Generating'):
                id_ = data_dict['q_id']
                instruction = data_dict['instruction']
                query_ids += id_
                label = data_dict['label']
                labels += label
                queries += data_dict['query']
                ranking_labels += data_dict['ranking_label']
                instructions += instruction
                generated_response = self.generate(data_dict['model_input'])
                responses += generated_response
        return query_ids, queries, instructions, responses, labels, ranking_labels
    
    def get_response(self):
        return '[/INST]' # TODO: this shouldn't be hard-coded

    def get_response_template_ids(self):
        response_template = self.get_response()
        return self.model.decoder_tokenizer.encode(response_template, add_special_tokens=False)
    
    def collate_fn(self, examples, eval=False):
        """
        Collates a batch of examples.

        Args:
            examples (list): batch from dataset
            eval (bool): Whether the function is being called for evaluation.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Collated batch of data.
        """
        from cocom.utils import add_memory_tokens_to_inputs

        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]
        query = [e['query'] for e in examples]

        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        for ex in examples:
            assert len(ex['doc']) == self.model.generation_top_k, f"Not all queries of the same number of docs: not supported here: {len(ex['doc'])} vs {self.model.generation_top_k}"

        docs = sum([example['doc'] for example in examples], []) # flatten all the docs for encoder input

        compressor_tokenizer = self.model.compr.tokenizer if self.model.compr else self.model.decoder_tokenizer

        assert compressor_tokenizer == self.model.decoder_tokenizer, 'Not supported yet'

        # Building encoder inputs, for each doc:
        inp_enc = [self.model.decoder_tokenizer.enc_token + self.model.decoder_tokenizer.bos_token + doc + self.model.decoder_tokenizer.eos_token for doc in docs]
        inp_enc = self.model.decoder_tokenizer(inp_enc, return_tensors='pt', padding="longest", max_length=self.context_max_length+3,
                                               truncation=True, add_special_tokens=False)

        # Getting the number of memory tokens to use for this batch
        num_mem_tokens = math.ceil((inp_enc['input_ids'].size(1)- 3) / self.model.compr_rate)


        inp_enc['input_ids'], inp_enc['attention_mask'] = add_memory_tokens_to_inputs(inp_enc['input_ids'],
                                                                                      inp_enc['attention_mask'],
                                                                                      num_mem_tokens,
                                                                                      self.model.decoder_tokenizer)

        assert num_mem_tokens == len(self.model.decoder_tokenizer.mem_tokens)
        mem_tokens_str = ''.join(self.model.decoder_tokenizer.mem_tokens)
        
        if self.model.sep:
            mem_tokens_str += self.model.decoder_tokenizer.sep_token

        # Internally, the model forward will concatenate the memory tokens from all documents for each query
        # We just need to leave some extra empty tokens when preparing the decoder inputs here:
        if eval:
            label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
            instr = [self.blend_prompt_and_memory_tokens(self.model.decoder_tokenizer, mem_tokens_str, query=q) for q in query]

            inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,
                                        truncation=True,  max_length=self.model_max_length)
        else:
            label = [e['label'] if isinstance(e['label'], str) else random.choice(e['label']) for e in examples]
            instr = [self.blend_prompt_and_memory_tokens(self.model.decoder_tokenizer, mem_tokens_str, query=q, label=e) for q, e in zip(query, label)]
            inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,
                                                   truncation=True, max_length=self.model_max_length)
            
            label_ids = prepare_labels(inp_dec["input_ids"], self.response_token_ids[1:], ignore_index=ignore_index)

        data_dict = {}
        if not eval:
            data_dict['label_ids'] =  label_ids

        model_input = {
            'enc_input_ids': inp_enc['input_ids'],
            'enc_attention_mask': inp_enc['attention_mask'],
            'dec_input_ids': inp_dec['input_ids'],
            'dec_attention_mask': inp_dec['attention_mask'],
        }
        
        data_dict.update({
            'model_input': model_input,
            'q_id': q_ids,
            'query': query,
            'instruction': instr,
            'label': label,
            'ranking_label': ranking_label,
        })
                        
        return data_dict

    def prediction_step(self, model, model_input, label_ids=None):
        # used during training
        output = model.forward(
            enc_input_ids=model_input['enc_input_ids'],
            enc_attention_mask=model_input['enc_attention_mask'],
            dec_input_ids=model_input['dec_input_ids'],
            dec_attention_mask=model_input['dec_attention_mask'],
            labels=label_ids
        )
            
        return output['logits'], output['loss']
    
    def blend_prompt_and_memory_tokens(self, tokenizer, mem_tokens: str, query: str, label: str=None):
        """
        Takes care of blending the prompt with the memory tokens:
        """        
        # proper name for "eval" call
        docs = mem_tokens * self.model.generation_top_k
        question = query
        
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": eval(self.prompt.user).replace(':\ ', ': ')}
        ]
        if label is not None:
            messages.append({"role": "assistant", "content": label})
            
        # todo: when training, should add_generation_prompt be set to True ?
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    
        return prompt


class LLMCocomOnlyDecoder(LLM):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 checkpoint_path: str,
                 context_max_length: int = 128,
                 max_doc_len: int = 100,
                 max_new_tokens: int = 128,
                 model_max_length: int = 1280,
                 prompt: str = None):
        """
        Uses the decoder of COCOM as an independent LLM (for evaluation purposes)
        This is just a reimplementation of LLM where in the init we extract the decoder from the cocom object
        checkpoint_path: path to a COCOM checkpoint
        context_max_length: maximum length for encoding documents
        max_new_tokens: maximum number of tokens for generation
        model_max_length: maximum length used in the final query (should be large enough)
        """
        from cocom.model import COCOM
        
        Generator.__init__(self, model_name=model_name, batch_size=batch_size)

        # Loading the cocom model:
        cocom = COCOM.from_pretrained(checkpoint_path)
        
        # We will only need this adapter if it exists
        if 'decoder_adapter' in cocom.adapter_keys:
            print('Activating decoder adapter.')
            cocom.decoder.set_adapter('decoder_adapter')
            
        cocom.eval()
        
        self.model = cocom.decoder
        self.tokenizer = cocom.decoder_tokenizer

        self.max_doc_len = max_doc_len # limit in words for documents.

        self.prompt = prompt
        self.context_max_length = context_max_length
        self.max_new_tokens = max_new_tokens
        self.model_max_length = model_max_length
        
        self.model.bfloat16()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token
