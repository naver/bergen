import torch
import math
import random
from models.generators.generator import Generator
from models.generators.cocom import COCOM, COCOMConfig
from utils import prepare_labels

random.seed(42)
def freeze_model(model):
    for _, param in model.named_parameters():
        param.requires_grad = False

class COCOMLLM(Generator):
    def __init__(self, 
                model_name = "cocom", 
                decoder_model_name="meta-llama/Llama-2-7b-chat-hf",
                max_new_tokens = 128, 
                quantization = 'no', 
                generation_top_k = 1, 
                sep = True,
                compr_model_name = "bert-base-uncased", 
                compr_rate = 64,
                compr_linear_type = 'concat',
                lora = False,
                query_dependant = False,
                training_form="both",
                context_max_length=512,
                test_mode="ft",
                **kwargs,
    ):
        if model_name == 'cocom':
            cfg = COCOMConfig(
                decoder_model_name=decoder_model_name,
                max_new_tokens=max_new_tokens,
                quantization=quantization,
                generation_top_k=generation_top_k,
                sep=sep,
                compr_model_name=compr_model_name,
                compr_rate=compr_rate,
                compr_linear_type=compr_linear_type,
                lora = lora
                )
            self.model = COCOM(cfg)
        else:
            self.model = COCOM.from_pretrained(model_name, ignore_mismatched_sizes=True)
            # if self.model.lora:
            #     self.model.decoder.merge_and_unload()
            self.model.sep = sep
            self.model.compr_rate = compr_rate
            self.model.generation_top_k = generation_top_k
            self.model.max_new_tokens = max_new_tokens
            print(self.model.compr_rate,self.model.generation_top_k,self.model.max_new_tokens)
        self.tokenizer=None

        self.training_form = training_form
        assert self.training_form in ['decoder', 'compressor', 'linear', 'both']
        self.test_mode = test_mode
        assert self.test_mode in ['ft', 'ae']
        if self.test_mode == 'ae':
            self.model.sep = False

        if self.training_form == 'decoder':
            freeze_model(self.model.compr.model)
        elif self.training_form == 'compressor':
            freeze_model(self.model.decoder)
        elif self.training_form == 'linear':
            freeze_model(self.model.compr.model)
            freeze_model(self.model.decoder)
    
        self.model_name = model_name
        self.query_dependant = query_dependant
        self.max_new_tokens = max_new_tokens
        self.context_max_length = context_max_length
        self.response_token_ids = self.get_response_template_ids()
        print ("context_max_length ",context_max_length)
        print("Response token ids")
        print(self.response_token_ids)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def generate(self, instr_tokenized):
        return self.model.generate(instr_tokenized, max_new_tokens=self.max_new_tokens)
        
    
    def get_response(self):
        #return '\n[/INST]\n'
        return "\n### Assistant:\n"
    

    def get_response_template_ids(self):
        response_template = self.get_response()
        return self.model.decoder_tokenizer.encode(response_template, add_special_tokens=False)
        

    def prediction_step(self, model, model_input, label_ids=None):
        output = model.forward(**model_input, labels=label_ids)
        return output['logits'], output['loss']
    
    def collate_fn(self, examples,  eval=False, **kwargs):
            ignore_index = -100
            q_ids = [e['q_id'] for e in examples]
            query = [e['query'] for e in examples]
  
            ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
            docs = sum([example['doc'] for example in examples], [])
            if self.model.compr is not None:
                # case bert-compressor
                if self.query_dependant:
                    #repeate query by generation_top_k times
                    query_combined = [q for q in query for _ in range(self.model.generation_top_k)]
                    docs = [q + "[SEP]" + d for q, d in zip(query_combined, docs)]
                if self.test_mode == 'ae':
                    inp_enc = self.model.compr.tokenizer(docs, return_tensors='pt', padding=True, max_length=self.context_max_length+3, truncation=True, pad_to_multiple_of=self.model.compr_rate)
                    if inp_enc['input_ids'].size(1) < self.context_max_length:
                        num_mem_tokens = math.ceil(inp_enc['input_ids'].size(1) / self.model.compr_rate)
                    else:
                        num_mem_tokens = math.ceil(self.context_max_length / self.model.compr_rate)
                else:
                    inp_enc = self.model.compr.tokenizer(docs, return_tensors='pt', padding=True, max_length=self.context_max_length, truncation=True, pad_to_multiple_of=self.model.compr_rate)
                    num_mem_tokens = math.ceil(inp_enc['input_ids'].size(1) / self.model.compr_rate)
            else:
                # case decoder-compressor
                # first add bos in the beginning of the input, eos in the end
                if self.query_dependant:
                    #repete query by generation_top_k times
                    query_combined = [q for q in query for _ in range(self.model.generation_top_k)]
                    docs = [q + self.model.decoder_tokenizer.sep_token + d for q, d in zip(query_combined, docs)]
                self.context_max_length=128
                inp_enc = [self.model.decoder_tokenizer.enc_token + self.model.decoder_tokenizer.bos_token + doc + self.model.decoder_tokenizer.eos_token for doc in docs]
                inp_enc = self.model.decoder_tokenizer(inp_enc, return_tensors='pt', padding=True, add_special_tokens=False, truncation=True, pad_to_multiple_of=self.model.compr_rate, max_length=self.context_max_length)
                num_mem_tokens = math.ceil(inp_enc['input_ids'].size(1) / self.model.compr_rate)
                #print (num_mem_tokens, inp_enc['input_ids'].size(1),self.model.compr_rate)
                # append memory tokens to the input
                mem_tokens = torch.full((inp_enc['input_ids'].size(0), num_mem_tokens), self.model.decoder_tokenizer.mem_token_id, dtype=torch.long)
                inp_enc['input_ids'] = torch.cat([inp_enc['input_ids'], mem_tokens], dim=1)
                inp_enc['attention_mask'] = torch.cat([inp_enc['attention_mask'], torch.ones(inp_enc['attention_mask'].size(0), num_mem_tokens)], dim=1)
            # input for decoder
            # [Padding][BOS][mem][INST]{question}?[/INST]
            mem_tokens = self.model.decoder_tokenizer.mem_token * num_mem_tokens
            #print(self.model.sep)
            if self.model.sep:
                mem_tokens += self.model.decoder_tokenizer.sep_token
            mem_tokens= mem_tokens *  self.model.generation_top_k
            print(mem_tokens, self.model.generation_top_k)#, mem_tokens *  self.model.generation_top_k )
            if eval:
                #for inference
                sysprompt="You are a helpful assistant. Your task is to extract relevant information from provided documents and to answer to questions as briefly as possible.\n"
                sysprompt_nodocs="You are a helpful assistant. Your task is to answer to questions as briefly as possible.\n"

                label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
                if self.test_mode == 'ae':
                    instr = [self.model.decoder_tokenizer.ae_token + self.model.decoder_tokenizer.bos_token + mem_tokens * self.model.generation_top_k for q in query]
                else:
                    #instr = [self.model.decoder_tokenizer.bos_token + mem_tokens * self.model.generation_top_k + '[INST]' + q + self.get_response() for q in query]
                    #instr = [ f"### System:\n {sysprompt}"+ '### User:\nBackground:' +mem_tokens * self.model.generation_top_k +" Question: " + q + "### Assistant:\n" for q in query]
                    instr = [ f"### System:\n {sysprompt}"+ '### User:\nBackground:' + mem_tokens  + " Question: " + q + "### Assistant:\n" for q in query]
                    print (instr)
                    #no docs
                    #instr = [ f"### System:\n {sysprompt_nodocs}"+ "### User:\n Question: " + q + "### Assistant:\n" for q in query]
                inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False,
                                        truncation=True,  max_length=768)
            else:
                if self.test_mode == 'ae':
                    # it's not possible to have ae mode for training
                    raise ValueError("AE mode is not possible for training")
                label = [e['label'] if isinstance(e['label'], str) else random.choice(e['label']) for e in examples]
     
                instr = [self.model.decoder_tokenizer.bos_token + mem_tokens * self.model.generation_top_k + '[INST]' + q + self.get_response() + e + self.model.decoder_tokenizer.eos_token  for q, e in zip(query, label)]
                inp_dec = self.model.decoder_tokenizer(instr, return_tensors='pt', padding="longest", add_special_tokens=False, truncation=True, max_length=768)
                label_ids = prepare_labels(inp_dec["input_ids"], self.response_token_ids[1:], ignore_index=ignore_index)
                # print the ones that has all -100
                for i, label_id in enumerate(label_ids):
                    if all([l == ignore_index for l in label_id]):
                        print(f"Warning: all -100 in label_ids {i}")
                        print(instr[i])
                        print(inp_dec["input_ids"][i])
                        print(label_id)
                    

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
           