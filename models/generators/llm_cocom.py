import torch
import random
from tqdm import tqdm
from models.generators.generator import Generator
from models.generators.llm import LLM
from torch.utils.data import DataLoader
from jinja2.exceptions import TemplateError


class LLMCocom(Generator):
    def __init__(self,
                 model_name: str,
                 batch_size: int,
                 checkpoint_path: str,
                 context_max_length: int = 128,
                 decoder_model_name: str = 'mistralai/Mistral-7B-Instruct-v0.2',
                 max_new_tokens: int = 128,
                 quantization='no',
                 model_max_length: int = 1280,
                 prompt: str = None,
                 lora_compressor: bool = False,
                 compr_rate: float = None,
                 compr_model_name: str = None,
                 compr_mode: str = 'last',
                 compr_mlp_hidden_dim: int = 1024,
                 compr_n_layers: int = None, # only useful for surgical mistral compressor,
                 compr_rms_norm: bool = False, # only useful for surgical mistral compressor
                 compr_use_mlp: bool = True,
                 attn_implementation: str = 'flash_attention_2',
                 query_dependent: bool = False,
                 device_map = 'auto',
                 save_generated_embeddings_path: str = None):
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
            self.model = COCOM.from_pretrained(checkpoint_path, device_map=device_map, attn_implementation=attn_implementation)
                
        else:
            cfg = COCOMConfig(
                decoder_model_name=decoder_model_name,
                max_new_tokens=128,
                quantization=quantization,
                compr_model_name=compr_model_name,
                compr_mode=compr_mode,
                compr_rate=compr_rate,
                compr_n_layers=compr_n_layers,
                compr_rms_norm=compr_rms_norm,
                compr_use_mlp=compr_use_mlp,
                compr_mlp_hidden_dim=compr_mlp_hidden_dim,
                lora=True,
                lora_compressor=lora_compressor,
                training_form='both_separately',
                lora_r=16,
                kbtc_training=False,
                optimize_mem_tokens=True,
                different_mem_tokens=True,
                device_map=device_map,
                attn_implementation=attn_implementation,
            )
            print('Creating brand new COCOM model:', cfg)
            self.model = COCOM(cfg)
            
        #self.model.eval()
        
        self.prompt = prompt
        self.context_max_length = context_max_length
        self.query_dependent = query_dependent
        if self.query_dependent:
            self.context_max_length += 64  # hard-coded at the moment
        self.max_new_tokens = max_new_tokens
        self.model_max_length = model_max_length
        self.save_generated_embeddings_path = save_generated_embeddings_path
        self.tokenizer = self.model.decoder_tokenizer
        if self.save_generated_embeddings_path is not None:
            print(f'I will save 100 doc embeddings at {self.save_generated_embeddings_path}')
        
    def generate(self, instr_tokenized, return_doc_embeddings: bool = False):
        """
        Nothing to do here, just convey to cocom since instr_tokenized went throught the collate_fn
        """
        device = next(self.model.parameters()).device
        instr_tokenized = {k: v.to(device) for k,v in instr_tokenized.items() if isinstance(v, torch.Tensor)}
        return self.model.generate(instr_tokenized, max_new_tokens=self.max_new_tokens, return_doc_embeddings=return_doc_embeddings)

    def eval(self, dataset):
        """
        We re-implement this method since we do not want to use the already existing "TokenizedDataset", we need
        to do the tokenization deeper in the process.
        dataset: returned by utils.prepare_dataset_from_ids
        """
        assert len(dataset) > 0, 'Empty dataset'
        self.model.eval() # setting eval mode

        # We get here the generation top k value, used in the encoder/decoder routine to batch encodings.
        # ideally we would convey it somewhere else, but we use asserts later on to continuously check it's unchanged
        example_docs = dataset[0]['doc']
        self.model.generation_top_k = len(example_docs)
        self.model.eval()

        dataloader = DataLoader(dataset, batch_size=self.batch_size,
                                    collate_fn=lambda l: self.collate_fn(l, eval=True), num_workers=4)

        responses, instructions, query_ids, queries, labels, ranking_labels = list(), list(), list(), list(), list(), list()
        doc_embeddings = []
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
                if self.save_generated_embeddings_path is not None:
                    generated_response, docs_embeddings = self.generate(data_dict['model_input'], return_doc_embeddings=True)
                    doc_embeddings.append(docs_embeddings.to('cpu'))
                else:
                    generated_response = self.generate(data_dict['model_input'], return_doc_embeddings=False)
                responses += generated_response
        
        if self.save_generated_embeddings_path is not None:
            doc_embeddings = torch.cat(doc_embeddings, 0) # of shape (dataset_length, top_k, n_mem, hidden_size)
            doc_embeddings = doc_embeddings[:min(len(doc_embeddings), 100)]
            torch.save(doc_embeddings, self.save_generated_embeddings_path)
        
        return query_ids, queries, instructions, responses, labels, ranking_labels
    
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
        batch_size = len(examples)

        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]
        query = [e['query'] for e in examples]

        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        for ex in examples:
            assert len(ex['doc']) == self.model.generation_top_k, \
                f"Not all queries of the same number of docs: not supported here: {len(ex['doc'])} vs {self.model.generation_top_k}"

        #### BULIDING ENCODER INPUTS ####
        docs = sum([example['doc'] for example in examples], []) # flatten all the docs for encoder input
        if self.query_dependent:
            # also add the flattened query to the input for the encoder
            flattened_query = sum([[q] * self.model.generation_top_k for q in query], [])
            inp_enc = self.model.prepare_encoder_inputs(texts=docs, q_texts=flattened_query, max_length=self.context_max_length)
        else:
            inp_enc = self.model.prepare_encoder_inputs(texts=docs, max_length=self.context_max_length, q_texts=None)
            
        enc_input_ids, enc_attention_mask = inp_enc['input_ids'], inp_enc['attention_mask']
        
        # input_ids are of shape (top_k * batch_size, enc_token_length)
        # We can reshape it to (batch_size, top_k, enc_token_length)
        # for proper batching (important with multi-gpu training)
        assert enc_input_ids.size(0) == self.model.generation_top_k * batch_size
        assert len(enc_input_ids.size()) == 2
        assert enc_attention_mask.size(0) == self.model.generation_top_k * batch_size
        assert len(enc_attention_mask.size()) == 2
        enc_input_ids = enc_input_ids.view(batch_size, self.model.generation_top_k, -1)
        enc_attention_mask = enc_attention_mask.view(batch_size, self.model.generation_top_k, -1)

        #### BUILDING DECODER INPUTS ####
        mem_tokens_str = ''.join(self.model.decoder_tokenizer.mem_tokens)
        
        if self.model.sep:
            mem_tokens_str += self.model.decoder_tokenizer.sep_token

        # Internally, the model forward will concatenate the memory tokens from all documents for each query
        # We just need to leave some extra empty tokens when preparing the decoder inputs here:
        if eval:
            label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
            instr = [self.blend_prompt_and_memory_tokens(self.model.decoder_tokenizer, mem_tokens_str, query=q)[0] for q in query]

            inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,
                                        truncation=True,  max_length=self.model_max_length)
        else:
            label = [e['label'] if isinstance(e['label'], str) else random.choice(e['label']) for e in examples]
                        
            instr, labels_start = zip(*[self.blend_prompt_and_memory_tokens(self.model.decoder_tokenizer, mem_tokens_str, query=q, label=e)
                                        for q, e in zip(query, label)])
            instr, labels_start = list(instr), list(labels_start)
                
            inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,
                                                   truncation=True, max_length=self.model_max_length)
            
            # sadly the tokenization with padding added some left padding, so we must update labels accordingly
            label_ids = inp_dec['input_ids'].clone()
            for i in range(len(label_ids)):
                # this counts the amount of left padding added to this particular item in the batch:
                left_padding_count = (inp_dec['attention_mask'][i] == 0).sum().item()  # This counts the number of padding tokens on the left
                # we do not count in the loss the padding elements and the question part:
                label_ids[i, :labels_start[i]+left_padding_count] = ignore_index

        data_dict = {}
        if not eval:
            #data_dict['labels'] =  label_ids
            return {
            'enc_input_ids': enc_input_ids,
            'enc_attention_mask': enc_attention_mask,
            'dec_input_ids': inp_dec['input_ids'],
            'dec_attention_mask': inp_dec['attention_mask'],
            'labels': label_ids
        }

        model_input = {
            'enc_input_ids': enc_input_ids,
            'enc_attention_mask': enc_attention_mask,
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
        Also returns, if a label is provided, the position of the first token index of the label (for loss comp later on)
        """        
        
        # proper names for "eval" call, don't remove these lines
        docs = mem_tokens * self.model.generation_top_k
        question = query
        
        # Prepare the messages with system and user roles
        messages = [
            {"role": "system", "content": self.prompt.system},
            {"role": "user", "content": eval(self.prompt.user).replace(':\ ', ': ')}
        ]

        # Attempt to apply the system role and catch if it's not supported
        try:
            # Handle the label
            label_start = None
            if label is not None:
                messages.append({"role": "assistant", "content": label})
                # Compute the start position of the label in the input_ids
                label_start = len(tokenizer.apply_chat_template(
                    messages[:-1], tokenize=True, add_generation_prompt=True, add_special_tokens=False))

            # Tokenize the full messages list, including system role
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(label is None))
            
        except TemplateError as e:
            # Catch the error related to system role and handle it (e.g. gemma)
            if "System role not supported" in str(e):
                # Remove system role and proceed with only the user role
                messages = [{"role": "user", "content": messages[0]['content'] + '\n' + messages[1]['content']}]
                
                label_start = None
                
                if label is not None:
                    messages.append({"role": "assistant", "content": label})
                    # Recompute the label_start without the system role
                    label_start = len(tokenizer.apply_chat_template(
                        messages[:-1], tokenize=True, add_generation_prompt=True, add_special_tokens=False))
                
                # Apply template again without system role
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=(label is None))
            else:
                # Re-raise the exception if it's unrelated to system role
                raise e

        return prompt, label_start
