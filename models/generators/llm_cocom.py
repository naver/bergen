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
                 compr_rate: float = None,
                 device_map = 'auto'):
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
            self.model = COCOM.from_pretrained(checkpoint_path, device_map=device_map)
        else:
            cfg = COCOMConfig(
                decoder_model_name=decoder_model_name,
                max_new_tokens=128,
                quantization=quantization,
                compr_model_name=None,
                compr_rate=compr_rate,
                lora=True,
                training_form='both_separately',
                lora_r=16,
                kbtc_training=False,
                optimize_mem_tokens=True,
                different_mem_tokens=True,
                device_map=device_map,
            )
            print('Creating brand new COCOM model:', cfg)
            self.model = COCOM(cfg)
            
        #self.model.eval()
        
        self.prompt = prompt
        self.context_max_length = context_max_length
        self.max_new_tokens = max_new_tokens
        self.model_max_length = model_max_length
                
    def generate(self, instr_tokenized):
        """
        Nothing to do here, just convey to cocom since instr_tokenized went throught the collate_fn
        """
        device = next(self.model.parameters()).device
        instr_tokenized = {k: v.to(device) for k,v in instr_tokenized.items() if isinstance(v, torch.Tensor)}
        return self.model.generate(instr_tokenized, max_new_tokens=self.max_new_tokens)

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
        batch_size = len(examples)

        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]
        query = [e['query'] for e in examples]

        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        for ex in examples:
            assert len(ex['doc']) == self.model.generation_top_k, \
                f"Not all queries of the same number of docs: not supported here: {len(ex['doc'])} vs {self.model.generation_top_k}"

        compressor_tokenizer = self.model.compr.tokenizer if self.model.compr else self.model.decoder_tokenizer
        assert compressor_tokenizer == self.model.decoder_tokenizer, 'Not supported yet'

        #### BULIDING ENCODER INPUTS ####
        docs = sum([example['doc'] for example in examples], []) # flatten all the docs for encoder input
        inp_enc = [self.model.decoder_tokenizer.enc_token + self.model.decoder_tokenizer.bos_token + doc + self.model.decoder_tokenizer.eos_token for doc in docs]
        inp_enc = self.model.decoder_tokenizer(inp_enc, return_tensors='pt', padding="longest", max_length=self.context_max_length+3, truncation=True, add_special_tokens=False)

        # Getting the number of memory tokens to use for this batch
        num_mem_tokens = 128 // self.model.compr_rate # TODO: should not be hard-coded like this.

        enc_input_ids, enc_attention_mask = add_memory_tokens_to_inputs(inp_enc['input_ids'],
                                                                                      inp_enc['attention_mask'],
                                                                                      num_mem_tokens,
                                                                                      self.model.decoder_tokenizer)
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
        assert num_mem_tokens == len(self.model.decoder_tokenizer.mem_tokens)
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
        
        # label_start = None
        
        # if label is not None:
        #     messages.append({"role": "assistant", "content": label})
        #     # Compute the start position of the label in the input_ids
        #     label_start = len(tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, add_special_tokens=False))

        # # todo: when training, should add_generation_prompt be set to True ?
        # prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        
        # return prompt, label_start
    
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
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            
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
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            else:
                # Re-raise the exception if it's unrelated to system role
                raise e

        return prompt, label_start


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
