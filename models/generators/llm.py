'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import random
import os
import json
import gc
import torch
import warnings

from peft import AutoPeftModelForCausalLM, PeftConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils import prepare_labels, left_pad
from models.generators.generator import Generator

random.seed(42)


class LLM(Generator):
    def __init__(self, 
                model_name: str = None,
                batch_size: int = 1, 
                max_new_tokens: int = 1, 
                max_doc_len: int = 100,
                max_length: int = None,
                prompt: str = None,
                quantization: str = None,
                gguf_file: str = None,
                attn_implementation: str = "flash_attention_2",
                local_path: bool = False,
                ):
        """
        :model_name: hf model name or path to a local checkpoint
        :max_new_tokens: how many tokens to generate at most
        :max_doc_len: documents are cropped to a maximum of max_doc_len words
        :gguf_file: specify to use a gguf_file (see from_pretrained)
        :local_path: forces only local reading (i.e. no hf download)
        """
        Generator.__init__(self,
                           model_name=model_name,
                           batch_size=batch_size,
                           max_new_tokens=max_new_tokens,
                           max_doc_len=max_doc_len,
                           max_length=max_length)
        # check type of gpu: if not A100 then change attn implementation to sdpa
        if "A100" not in torch.cuda.get_device_name(torch.cuda.current_device):
            attn_implementation="sdpa"
            
        self.quantization = quantization
        
         # get tokenizer or lora adapter if it exists else use models' tokenizer
        if quantization == "no":
            tokenizer_name = self.model_name
            model_class = AutoModelForCausalLM
        else:
            try:
                config = PeftConfig.from_pretrained(model_name)
                tokenizer_name = config.base_model_name_or_path
                model_class = AutoPeftModelForCausalLM
                print(f"Found peft config for {model_name}")
            except:
                warnings.warn(f"Could not find PeftConfig for {model_name}. Using regular model.")
                tokenizer_name = self.model_name
                model_class = AutoModelForCausalLM
                
        print(f"Tokenizer used: {tokenizer_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, gguf_file=gguf_file)
        except:
            config_dict = os.path.join(tokenizer_name, 'config.json')
            with open(config_dict, 'r') as f:
                config = json.load(f)
            tokenizer_name = config['_name_or_path']
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, gguf_file=gguf_file)

        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.bos_token

        if quantization == "int8":
            quant_config = BitsAndBytesConfig(
                llm_int8_enable_fp32_cpu_offload=True
            )
            self.model = model_class.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                local_files_only=local_path,
            )

        elif quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
            )

            self.model = model_class.from_pretrained(
                self.model_name,
                quantization_config=quant_config,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map='auto',
                local_files_only=local_path,
            )
        else:
            self.model = model_class.from_pretrained(
                self.model_name,
                device_map='auto',
                gguf_file=gguf_file,
            )

        self.model = self.model.bfloat16()
        self.model.eval()
        self.model.config.pretraining_tp = 1
        self.prompt = prompt

    def generate(self, instr_tokenized):
        input_ids = instr_tokenized['input_ids'].to(self.model.device)
        attention_mask = instr_tokenized['attention_mask'].to(self.model.device)
        output_ids = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
        )

        prompt_len = instr_tokenized['input_ids'].size(1)
        generated_ids = output_ids[:, prompt_len:]
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return decoded
        
    def __del__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def collate_fn(self, examples: list[dict], eval: bool = False, **kwargs):
        ignore_index = -100
        q_ids = [e['q_id'] for e in examples]

        input_ids_list = [e["tokenized_input"]["input_ids"][0] for e in examples]
        attention_mask_list = [e["tokenized_input"]["attention_mask"][0] for e in examples]

        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        instr = [e["formatted_instruction"] for e in examples]

        # NB: we apply padding after tokenization to be able to sort batches by size beforehand
        # Determine the maximum sequence length from input_ids
        max_length = max(len(ids) for ids in input_ids_list)
        
        input_ids_tensor = torch.stack([left_pad(ids, max_length, self.tokenizer.pad_token_id) for ids in input_ids_list])
        attention_mask_tensor = torch.stack([left_pad(mask, max_length, 0) for mask in attention_mask_list])
        
        model_input = {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
        }
        
        data_dict = {}
        
        # prepare labels: only for training 
        if not eval:
            label_ids = input_ids_tensor.clone()
            for i in range(len(label_ids)):
                assert examples[i]['label_start_index'] is not None
                # Count the number of padding tokens on the left
                left_padding_count = (attention_mask_tensor[i] == 0).sum().item()
                # todo: handle case where left padding count is too large or we cropped the label no ?
                # In the label there is only tokens after position padding_count + label_start_idnex:
                label_ids[i, :examples[i]['label_start_index']+left_padding_count + 1] = ignore_index
                
                # Here we can assert that the label_ids, when decoded, is included in one of the original labels
                original_labels = label[i]
                recovered_label = self.tokenizer.decode(label_ids[label_ids != ignore_index], skip_special_tokens=True).strip()
                # print('######YO check this out:', original_labels, 'VS', recovered_label, '###END OF YO')
                
                label_found = False
                for original_label in original_labels:
                    if recovered_label in original_label:
                        label_found = True
                if not label_found:
                    assert False, f"###### <{recovered_label}> NOT INCLUDED IN <{original_labels}>"
                 
            model_input['labels'] =  label_ids
            return model_input

        data_dict.update({
            'model_input': model_input,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        })

        return data_dict
