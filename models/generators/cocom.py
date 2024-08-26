from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PretrainedConfig, AutoModel
from accelerate import init_empty_weights
import torch
import math 
from peft import get_peft_model, LoraConfig, TaskType


class Compressor(torch.nn.Module):
    def __init__(self, compr_model_name, compr_rate, compr_linear_type, decoder_hidden_size):
        super().__init__()
        # init model
        self.model_name = compr_model_name
        self.model = AutoModel.from_pretrained(compr_model_name, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(compr_model_name, use_fast=True)
        self.compr_rate = compr_rate
        self.compressing_mode = compr_linear_type

        if self.compressing_mode == 'concat':
            self.linear = torch.nn.Linear(self.model.config.hidden_size*self.compr_rate, decoder_hidden_size)
        elif self.compressing_mode in ['cls', 'mean', 'sep']:
            self.linear = torch.nn.Linear(self.model.config.hidden_size, decoder_hidden_size)
        self.linear = self.linear.bfloat16()

    def forward(self, input_ids, attention_mask):
        segment_compress_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        num_embs = math.ceil(input_ids.size(1) / self.compr_rate)

        all_hidden_states_emb = list()
        if self.compressing_mode == 'concat':
            for segment_idx in range(num_embs):
                start_idx = segment_idx * self.compr_rate
                end_idx = (segment_idx + 1) * self.compr_rate
                hidden_state = segment_compress_outputs.hidden_states[-1][:, start_idx:end_idx, :]
                hidden_state_concat = torch.flatten(hidden_state, start_dim=1) #batch_size, hidden_state_dim * compression_rate
                all_hidden_states_emb.append(hidden_state_concat)


        elif self.compressing_mode == "mean":
            for segment_idx in range(num_embs):
                start_idx = segment_idx * self.compr_rate
                end_idx = (segment_idx + 1) * self.compr_rate
                hidden_state = segment_compress_outputs.hidden_states[-1][:, start_idx:end_idx, :]
                # Apply mean pooling to get the final embedding for the segment
                all_hidden_states_emb.append(hidden_state)


        all_hidden_states_emb_cat = torch.stack(all_hidden_states_emb, dim=1)
        transformed_embeds = self.linear(all_hidden_states_emb_cat)
        

        if self.compressing_mode == "mean":
            transformed_embeds = torch.mean(transformed_embeds, dim=2)


        return  transformed_embeds


class COCOMConfig(PretrainedConfig):

    model_type = "COCOM"
    def __init__(self,
                decoder_model_name="meta-llama/Llama-2-7b-chat-hf",
                quantization = 'int4', 
                generation_top_k = 1, 
                sep = False,
                compr_model_name = "bert-base-uncased", 
                compr_rate = 64,
                compr_linear_type = 'concat',
                lora = True,
                 **kwargs):
        super().__init__(**kwargs)

        self.decoder_model_name = decoder_model_name
        self.quantization = quantization
        self.generation_top_k = generation_top_k
        self.sep = sep
        self.compr_model_name = compr_model_name
        self.compr_rate = compr_rate
        self.compr_linear_type = compr_linear_type
        # lora could be a boolean or a str
        self.lora = lora
        if lora == "True":
            self.lora = True
        elif lora == "False":
            self.lora = False
        self.sep=True
        print(self.sep)
        

class COCOM(PreTrainedModel):
    config_class = COCOMConfig
    def __init__(self, cfg):
        super().__init__(cfg)
        # define models
        if cfg.quantization == "no":
            with init_empty_weights():
                self.decoder = AutoModelForCausalLM.from_pretrained(
                    cfg.decoder_model_name, 
                    torch_dtype=torch.bfloat16,
                    use_flash_attention_2=True, 
                    #attn_implementation='sdpa',
                    resume_download=True,
                    low_cpu_mem_usage = True,
                    trust_remote_code=True,
                    )
        elif cfg.quantization == "int4":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype='bfloat16',
                low_cpu_mem_usage = True,
            )
            self.decoder = AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name, 
                quantization_config=quant_config,
                attn_implementation="flash_attention_2", 
                torch_dtype=torch.bfloat16,
                resume_download=True,
                low_cpu_mem_usage = True,
                trust_remote_code=True,
            )
        elif cfg.quantization == "int8":
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
                bnb_4bit_compute_dtype='bfloat16',
                low_cpu_mem_usage = True,
            )
            self.decoder = AutoModelForCausalLM.from_pretrained(
                cfg.decoder_model_name,
                quantization_config=quant_config,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                resume_download=True,
                low_cpu_mem_usage = True,
                trust_remote_code=True,
            )
        else:
            raise NotImplementedError()
        
        if cfg.compr_model_name is not None:
            self.compr = Compressor(cfg.compr_model_name, cfg.compr_rate, cfg.compr_linear_type, self.decoder.config.hidden_size)
            #self.compr_pad_token_id = self.compr.pad_token_id
        else:
            self.compr = None
        
        self.lora = cfg.lora
        if cfg.lora:
            peft_config = LoraConfig(
                        task_type="CAUSAL_LM",
                        r=16,
                        lora_alpha=32,
                        target_modules='all-linear',
                        lora_dropout=0.1,
                    )
            self.decoder = get_peft_model(self.decoder, peft_config)
            self.decoder.print_trainable_parameters()  


        self.decoder_tokenizer = AutoTokenizer.from_pretrained(cfg.decoder_model_name, use_fast=True, padding_side='left')

        # define special tokens
        self.decoder_tokenizer.add_special_tokens({'additional_special_tokens': ['<MEM>', '<AE>', '<ENC>', '<SEP>']})
        self.decoder_tokenizer.mem_token = '<MEM>'
        self.decoder_tokenizer.ae_token = '<AE>'
        self.decoder_tokenizer.enc_token = '<ENC>'
        self.decoder_tokenizer.sep_token = '<SEP>'
        self.decoder_tokenizer.mem_token_id = self.decoder_tokenizer.convert_tokens_to_ids('<MEM>')
        self.decoder_tokenizer.ae_token_id = self.decoder_tokenizer.convert_tokens_to_ids('<AE>')
        self.decoder_tokenizer.sep_token_id = self.decoder_tokenizer.convert_tokens_to_ids('<SEP>')
         # if pad token ecist then use pad token, othrwise bos token
        if self.decoder_tokenizer.pad_token_id is None:
            self.decoder_tokenizer.pad_token_id = self.decoder_tokenizer.bos_token_id

        # resize the tokenizer embedding
        
        # define special tokens
        self.decoder.resize_token_embeddings(len(self.decoder_tokenizer))
        self.decoder.generation_config.top_p=None
        self.decoder.generation_config.temperature=None
        
        self.compr_model_name = cfg.compr_model_name
        # other settings
        self.generation_top_k = cfg.generation_top_k
        self.sep = cfg.sep
        self.compr_rate = cfg.compr_rate
    def compress_and_replace_emb(self, enc_input_ids, enc_attention_mask, dec_input_ids):
        indices = range(0, enc_input_ids.size(0) + 1, self.generation_top_k)
        if self.compr:
            compressed_embs = self.compr(enc_input_ids, enc_attention_mask)
            input_embeds = self.replace_embeddings(compressed_embs, dec_input_ids, indices)
        else:
            compressed_embs = self.compr_decoder(enc_input_ids, enc_attention_mask)
            input_embeds = self.replace_embeddings(compressed_embs, dec_input_ids, indices)
        return input_embeds
    
    def compr_decoder(self, input_ids, attention_mask):
        emb = self.decoder(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        mask = input_ids == self.decoder_tokenizer.mem_token_id
        return emb[mask].reshape(emb.size(0), -1, emb.size(-1))
    

    def replace_embeddings(self, compressed_embs, dec_input_ids, indices):
        # Embed the decoder input
        inputs_embeds = self.decoder.get_input_embeddings()(dec_input_ids)
        num_embs = compressed_embs.size(1)
        if self.sep:
            slot_len = num_embs + 1
        else:
            slot_len = num_embs
        # get first mem_token inidices
        first_mem_token_indices = torch.argmax((dec_input_ids == self.decoder_tokenizer.mem_token_id).int(), dim=1)
        batch_size = inputs_embeds.size(0)
        # for each example in batch
        for i in range(batch_size):
            for j in range(indices[i], indices[i + 1]):
                start_idx = first_mem_token_indices[i].item() + (j-indices[i]) * slot_len
                inputs_embeds[i, start_idx:start_idx + num_embs, :] = compressed_embs[j]
        return inputs_embeds


    def forward(self, 
        enc_input_ids: torch.LongTensor = None,
        enc_attention_mask: torch.LongTensor = None,
        dec_input_ids : torch.LongTensor = None, 
        dec_attention_mask: torch.LongTensor = None,
        labels: torch.LongTensor = None, 
        ):
        inputs_embeds = self.compress_and_replace_emb(enc_input_ids, enc_attention_mask, dec_input_ids)
        decoder_outputs = self.decoder(inputs_embeds=inputs_embeds, attention_mask=dec_attention_mask, labels=labels)
        return {"loss": decoder_outputs.loss, "logits": decoder_outputs.logits}


        
    def generate(self, model_input, max_new_tokens=128):
        enc_input_ids, enc_attention_mask, dec_input_ids, dec_attention_mask = model_input['enc_input_ids'], model_input['enc_attention_mask'], model_input['dec_input_ids'], model_input['dec_attention_mask']
        inputs_embeds = self.compress_and_replace_emb(enc_input_ids.to('cuda'), enc_attention_mask.to('cuda'), dec_input_ids.to('cuda'))


        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds.to("cuda"), 
            attention_mask=dec_attention_mask.to("cuda"),
            do_sample=False,
            top_p=None,
            max_new_tokens=max_new_tokens,
            )

        #output_ids = self.perform_generation(inputs_embeds.to('cuda'), dec_attention_mask.to('cuda'), max_new_tokens.to('cuda'))
        decoded = self.decoder_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        print(decoded)
        return decoded