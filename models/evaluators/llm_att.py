from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from models.generators.llm_cocom import LLMCocom
from tqdm import tqdm
import torch
import gc
import re
import numpy as np
from hydra.utils import instantiate
from torch.distributions import Categorical
from difflib import SequenceMatcher
from collections import defaultdict


class LLM_att():
    def __init__(self, generator_config, prompt):
        generator_config['init_args']['attn_implementation'] = 'sdpa'
        print(generator_config)
        self.llm = instantiate(generator_config['init_args'], prompt=prompt)
        
        self.embeddings_path = None
        if 'save_generated_embeddings_path' in generator_config['init_args']:
            self.embeddings_path = generator_config['init_args']['save_generated_embeddings_path'].replace('/tmp_', '/')
            print(f'Loading embeddings at {self.embeddings_path}')
            self.embeddings = torch.load(self.embeddings_path)
        
        #breakpoint()
        #self.llm = Generate(**generator_config, prompt=prompt, flash_att=False) if generator_config != None else None   

    def collate_fn(self, sample):
        #decompotes prompt into different subparts, and keep trace of subparts position (to quantify attention at these subparts)
        instr_subset = {}
        for prompt_el in ['system', 'context', 'system_without_docs', "user", 'user_without_docs', 'language_instruction']:
            #prompt_val = str(self.llm.model.prompt._content[prompt_el])
            try:
                prompt_val = str(self.llm.prompt[prompt_el])
            except:                    
                continue
            prompt_pos = sample['instruction'].find(prompt_val)
            prompt_el = prompt_el.replace("user", "context").replace("_without_docs", "").replace('language_instruction', 'system')
            if prompt_pos > 0:
                instr_subset[(prompt_pos, prompt_pos+len(prompt_val))] = prompt_el
            else:
                match = SequenceMatcher(None, sample['instruction'], prompt_val).find_longest_match()    
                if match.b <3:                
                    instr_subset[(match.a, -1)] = prompt_el
        instr_subset[(0,1)] = "bos"        
        qpos = sample['instruction'].find(sample['question'])
        if qpos>0:
            instr_subset[(qpos, qpos+len(sample['question']))] = 'question'
        substrings = []
        substrings_types = []
        start_pos = 0
        i = 0
        curr_label = 'bos'
        for pos in sorted(instr_subset):
            if pos[0]>=start_pos:
                substrings.append(sample['instruction'][start_pos:pos[0]])
                substrings_types.append(curr_label)
            if pos[1]>0:
                substrings.append(sample['instruction'][pos[0]:pos[1]])   
                substrings_types.append(instr_subset[pos])   
                start_pos = pos[1]
                curr_label = 'other'
            else:
                start_pos = pos[0]
                curr_label  = instr_subset[pos]
        
        substrings.append(sample['instruction'][start_pos:-1])
        substrings_types.append(curr_label)
        
        print('o')
        print(substrings)
        print(substrings_types)
        
        tokenized = self.llm.tokenizer(substrings+[sample['candidate']], is_split_into_words=True, add_special_tokens=False, return_tensors="pt")

        #tokenized_tensor = self.llm.tokenizer(substrings+[sample['candidate']], add_special_tokens=False, )
        prompt_tokenized = self.llm.tokenizer(substrings, is_split_into_words=True, add_special_tokens=False, return_tensors="pt")
        #breakpoint()
        prompt_len = prompt_tokenized.input_ids.size(1)
        return tokenized, prompt_len, substrings_types
    
    @torch.no_grad()
    def __call__(self, predictions, references, questions, instructions):
        def get_att_var(attention):
            # equation 2 from https://arxiv.org/pdf/2205.10828.pdf
            gen_len, prefix_len = attention.shape
            prefix_pos = torch.arange(0, prefix_len).to('cuda')
            mu = torch.matmul(prefix_pos.float(), attention.transpose(1,0).float())
            x = [torch.matmul(attention[i,:].float(), (mu[i]-prefix_pos)**2) for i in range(gen_len)]
            var = torch.mean(torch.stack(x))
            #for i in range(0, gen_len):
            #    var += torch.sum(attention[:,i]*(mu[i]-prefix_pos)**2)
            return var
        def get_att_entropy(attention):       
            """
            reflects how  peaky is the attention over prefix tokens, averaged over all generation tokens 
            higher values --> attention is more concentrated, lower values --> attention is more distributede
            https://aclanthology.org/I17-1004.pdf
            """     
            gen_len, prefix_len = attention.shape
            #need to normalize att over prefix to make it proper distribution
            attn_norm = attention/torch.sum(attention, axis=1, keepdim=True)
            entr_per_token = torch.sum(- attn_norm*torch.log(attn_norm), axis=1)
            return torch.mean(entr_per_token)

        def get_att_confidence(attention):  
            """
            maximum attention on the previx token averaged across all generated tokens 
            (could be lower for longer generations)
            """
            gen_len, prefix_len = attention.shape
            conf = torch.mean(torch.max(attention, axis=1)[0])

            return conf

        def get_att_coverage_conf(attention):  
            """
            maximum attention on the prefix token summed across all generated tokens 
            (should not penalize longer generations)
            """
            gen_len, prefix_len = attention.shape
            conf = torch.sum(torch.max(attention, axis=1)[0])

            return conf
        def get_att_coverage1(attention):  
            """
            |I| - prefix tokens length, |J| - generation length
            coverage = \avg_{j \in J} (\sum_{i \in I} \alpha_{ij})^2
            - reflects how much attention each generated token puts on the prefix, averaged across all generated tokens 
            taken from  https://arxiv.org/pdf/2105.14940
            """
        
            gen_len, prefix_len = attention.shape
            cov = torch.mean(torch.sum(attention, axis=1)**2)
            return cov

        def get_att_coverage(attention):  
            """
            |I| - prefix tokens length, |J| - generation lenght
            coverage = \sum_{i \in I} (\sum_{j \in J} \alpha_{ij})^2
            - reflects how much attention each token in the prefix has recieved from generated tokens
            summed over all prefix tokens --> overall coverage
            """
        
            gen_len, prefix_len = attention.shape
            cov = torch.sum(torch.sum(attention, axis=0)**2)
            return cov
        #i=0
        #batch_input_ids = instr_tokenized['input_ids'][i:i+self.llm.batch_size].to('cuda')
        #batch_attention_masks = instr_tokenized['attention_mask'][i:i+self.llm.batch_size].to('cuda')
        #instr_batch = instrs[i:i+self.llm.batch_size]
        #gen = self.llm.model.model.generate(input_ids=batch_input_ids, attention_mask=batch_attention_masks, do_sample=False, output_attentions=True, return_dict_in_generate=True)

        """
        def compute_att_variability(attentions):
            x = torch.div(torch.transpose(atts, 1,0), torch.sum(atts, axis=1))            
            ent 
        #output_ids = self.model.generate(**instr_tokenized.to('cuda'), max_new_tokens=self.max_new_tokens, max_length = self.max_length, do_sample=False)
        output = self.model.generate(**instr_tokenized.to('cuda'), max_new_tokens=self.max_new_tokens, max_length = self.max_length, do_sample=False,output_attentions=True, return_dict_in_generate=True)
        output_ids = output['sequences']
        full_attentions = output['attentions']
        prompt_toks = [self.tokenizer.decode(i) for  i in instr_tokenized['input_ids'][0]]        
        prompt_len = instr_tokenized.input_ids.size(1)     
        generated_ids = output_ids[:, prompt_len:]
        gen_toks  = [self.tokenizer.decode(i) for  i in generated_ids[0]]
        
        layer=-1
        nb_layers = len(full_attentions[0])
        #avg across heads 
        step = 3
        fig, axs = plt.subplots(1, int(nb_layers/step)+1, figsize=(50,50))            
        i = 0

        for layer in range(0, nb_layers, step):
            attentions = [torch.mean(att[layer], axis=1).squeeze(axis=1) for att in full_attentions]     
            #TODO : renormalize attention on prompt (will not sum to 1 for more than prompt_len tokens)

            prompt_to_gen_att = [attentions[j+1][0, :prompt_len] for j in range(generated_ids.shape[-1]-1)]
            atts = torch.stack(prompt_to_gen_att).float().cpu().numpy()
            df = pd.DataFrame(atts.transpose(), columns=gen_toks[1:], index=prompt_toks)
            sns.heatmap(df, ax=axs[i], cmap="crest")
            axs[i].set_title(f"Layer {layer}")
            i = i+1
        
        plt.savefig(f'plot_{self.model_name.split("/")[-1].replace("-","_")}.png')
        plt.clf()
        decoded = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        """
        assert len(predictions) == len(references) == len(questions) == len(instructions)
        examples = [{'question': questions[i], 'candidate': predictions[i], 'instruction': instructions[i]}  for i in range(len(predictions))]
        batch_size = self.llm.batch_size
        batch_size = 1
        layer = -1
        scores = []
        samples=100
        if samples == -1:
            samples = len(examples)
        for j in tqdm(range(0, min(len(examples), samples)), desc=' Compute attention-based metrics'):
        #for i in tqdm(range(0, len(examples)), desc=' Compute attention-based metrics'):
            # Extract batch
            batch_inputs, prompt_len, input_types  = self.collate_fn(examples[j])
            
            batch_input_ids = batch_inputs['input_ids'].to('cuda')
            batch_attention_masks = batch_inputs['attention_mask'].to('cuda')
            
            # For cocom models, we need the documents embeddings, we'll get them from here:
            if isinstance(self.llm, LLMCocom):
                compressed_embs = self.embeddings[j].to('cuda') # of shape (n_docs, n_tok, hidden_state)
                indices = range(0, compressed_embs.size(0) + 1, compressed_embs.size(0)) 
                inputs_embeds = self.llm.model.replace_embeddings(compressed_embs, batch_input_ids, indices)
                # Switch adapter if we are training two different ones:
                if 'decoder_adapter' in self.llm.model.adapter_keys:
                    self.llm.model.decoder.set_adapter('decoder_adapter')
                output = self.llm.model.decoder.generate(inputs_embeds=inputs_embeds,
                                                   attention_mask=batch_attention_masks,
                                                   max_new_tokens=1,
                                                   do_sample=False,
                                                   output_attentions=True,
                                                   output_hidden_states=True,
                                                   return_dict_in_generate=True)

            #breakpoint()
            else:
                #breakpoint()
                output = self.llm.model.generate(input_ids=batch_input_ids,
                                                 attention_mask=batch_attention_masks,
                                                 max_new_tokens=1,
                                                 do_sample=False,
                                                 output_attentions=True,
                                                 output_hidden_states=True,
                                                 return_dict_in_generate=True)
            #hidden states: (batch_size, layers, [bsize, seq_len, hidden_size])
            sequences = output['sequences']
            generated_ids = sequences[:, prompt_len:-1]
            #decoded = self.llm.model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            att_by_cat = defaultdict(float)
            if isinstance(self.llm, LLMCocom):
                layers = list(range(1, self.llm.model.decoder.config.num_hidden_layers, 3)) + [-1]    
            else:
                layers = list(range(1, self.llm.model.config.num_hidden_layers, 3)) + [-1]
            for layer in layers:
                attentions = output['attentions'][0][layer][0]                                        
                #avg across heads
                attentions = torch.mean(attentions, axis=0).squeeze(axis=0)# for att in full_attentions[0]
                hidden_states = output['hidden_states'][0][layer-1][0]
                if layer == -1:
                    layer = "last"                
                hidden_states_norm = torch.norm(hidden_states, dim=1)
                prompt_hidden_states = hidden_states_norm[:prompt_len]
                #gen_hidden_states = hidden_states_norm[prompt_len:]
                #prompt_to_gen_att_mh = full_attentions[:, prompt_len:, :prompt_len]
                prompt_to_gen_att = attentions[prompt_len:, :prompt_len] 
                #att_by_cat['att_prefix'] = torch.mean(torch.sum(prompt_to_gen_att, axis=1)).float().to('cpu').numpy()
                for i, cat in enumerate(input_types):
                    if not i+1 in batch_inputs.word_ids():
                        continue  
                    span = batch_inputs.word_to_tokens(i+1)
                    if span.start >= prompt_len:
                        continue
                    att_by_cat[f"att_{layer}_{cat}"] += torch.mean(torch.sum(prompt_to_gen_att[:, span.start:span.end], axis=1)).float().to('cpu').numpy()
                    att_by_cat[f"att_{layer}_{cat}_norm"] += torch.mean(torch.matmul(prompt_hidden_states[span.start:span.end], prompt_to_gen_att[:, span.start:span.end].t())).float().to('cpu').numpy()
                    att_by_cat[f"cov_{layer}_{cat}"] += get_att_coverage(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                    att_by_cat[f"cov1_{layer}_{cat}"] += get_att_coverage1(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()                    
                    att_by_cat[f"entropy_{layer}_{cat}"] += get_att_entropy(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                    att_by_cat[f"conf_{layer}_{cat}"] += get_att_confidence(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                    att_by_cat[f"conf1_{layer}_{cat}"] += get_att_coverage_conf(prompt_to_gen_att[:, span.start:span.end]).float().to('cpu').numpy()
                entropy= get_att_entropy(prompt_to_gen_att).float().to('cpu').numpy()
                entropy_nobos= get_att_entropy(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f"att_{layer}_prefix"] = np.sum([att_by_cat[x] for x in att_by_cat if not "norm" in x and f'{layer}' in x])
                att_by_cat[f"att_{layer}_prefix_norm"] = np.sum([att_by_cat[x] for x in att_by_cat if "norm" in x and f'{layer}' in x])
 
                #att_by_cat[f'var_{layer}'] += get_att_var(prompt_to_gen_att).float().to('cpu').numpy()
                #att_by_cat[f'var_{layer}_nobos'] += get_att_var(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f'entropy_{layer}'] += entropy     
                att_by_cat[f'entropy_{layer}_nobos'] += entropy_nobos
                att_by_cat[f'coverage_{layer}'] += get_att_coverage(prompt_to_gen_att).float().to('cpu').numpy()
                att_by_cat[f'coverage_{layer}_nobos'] += get_att_coverage(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f'conf_{layer}'] += get_att_confidence(prompt_to_gen_att).float().to('cpu').numpy()
                att_by_cat[f'conf_{layer}_nobos'] += get_att_confidence(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                att_by_cat[f'conf1_{layer}'] += get_att_coverage_conf(prompt_to_gen_att).float().to('cpu').numpy()
                att_by_cat[f'conf1_{layer}_nobos'] += get_att_coverage_conf(prompt_to_gen_att[:, 1:]).float().to('cpu').numpy()
                
                del attentions
                del hidden_states
                del prompt_hidden_states
                del prompt_to_gen_att
                torch.cuda.empty_cache()
                gc.collect()                    
            del output
            del batch_attention_masks
            del batch_input_ids
            del batch_inputs
            torch.cuda.empty_cache()
            gc.collect()
            scores.append(att_by_cat)    
            
        torch.cuda.empty_cache()
        return {cat: np.mean([score[cat] for score in scores]) for cat in att_by_cat.keys()}, scores