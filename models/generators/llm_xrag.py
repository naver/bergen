'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import AutoTokenizer
import torch
from models.generators.generator import Generator
import random
from transformers import StoppingCriteria
from models.generators.xrag import XMistralForCausalLM, SFR, XMixtralForCausalLM
import transformers
from typing import List


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ) -> None:
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence
        self.sequence_ids = tokenizer.encode(sequence, add_special_tokens=False)
        # print(sequence, self.sequence_ids)
        # we look back for 2 more tokens than it takes to encode our stop sequence
        # because tokenizers suck, and a model might generate `['\n', '\n']` but our `sequence` is `['\n\n']`
        # and we don't want to mistakenly not stop a generation because our
        # (string) stop sequence was output in a different tokenization

        # NOTE: there is a minor danger that this will end up looking back 2 tokens into the past, into the inputs to the model,
        # and stopping generation immediately as a result. With only 2 extra tokens of lookback, this risk is minimized
        # Additionally, in lookback_ids_batch we should prevent ever looking back into the inputs as described.
        self.sequence_id_len = len(self.sequence_ids) + 2
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length :]

        lookback_ids_batch = lookback_ids_batch[:, -self.sequence_id_len :]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker

## copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/cb22e5028a6e40f409a539cbdd87194fd5e2570c/lm_eval/models/utils.py#L248
def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    initial_decoder_input_length: int,
    batch_size: int,
    stop_sequences: List[str] = ['\n', '.', ','],
    ) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )

random.seed(42)
class LLM(Generator):
    def __init__(self, 
                model_name=None, 
                max_new_tokens=1, 
                max_doc_len=100,
                max_length=None,
                prompt=None,
                quantization=None,
                attn_implementation="flash_attention_2",
                generation_top_k=None
                 ):

        # device_index = Accelerator().process_index
        # device_map = {"": device_index}

        self.max_length = max_length
        self.model_name = model_name
        self.max_doc_len = max_doc_len
        self.quantization = quantization


        if 'moe' in model_name:
            model_class = XMixtralForCausalLM
        else:
            model_class = XMistralForCausalLM


        # load decoder llm 

        self.model = model_class.from_pretrained(
        model_name,
        torch_dtype = torch.bfloat16,
        low_cpu_mem_usage = True, 
        load_in_4bit=True,
        device_map="auto")
        self.tokenizer  = AutoTokenizer.from_pretrained(model_name, add_eos_token=False, use_fast=False, padding_side='left')
        
        # set xrag token
        self.XRAG_TOKEN = "<xRAG>"
        self.model.set_xrag_token_id(self.tokenizer.convert_tokens_to_ids(self.XRAG_TOKEN))

        # load retriever
        retriever_name_or_path = "Salesforce/SFR-Embedding-Mistral"
        self.retriever = SFR.from_pretrained(retriever_name_or_path,torch_dtype = torch.bfloat16, load_in_4bit=True, device_map="auto").eval()
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(retriever_name_or_path)


        # self.model.merge_and_unload()
        #self.model.config.use_cache = False
        self.model = self.model.bfloat16()
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.prompt = prompt

    def get_response(self):
        return ''

    def prediction_step(self, model, model_input, label_ids=None):
        pass

    def generate(self, inp):

        docs, instructions = inp

        retriever_input = self.retriever_tokenizer(docs, max_length=180, padding=True, truncation=True, return_tensors='pt').to('cuda')
        with torch.no_grad():
            doc_embeds = self.retriever.get_doc_embedding(input_ids=retriever_input.input_ids, attention_mask=retriever_input.attention_mask)

        input_xrag = self.tokenizer(instructions,return_tensors='pt', padding=True, truncation=True).to('cuda') #padding=True,truncation=True
        if input_xrag['input_ids'].shape[0] != 1:
            raise NotImplementedError('Stopping criteria only works for batch size 1!')
        stopping_criteria = stop_sequences_criteria(self.tokenizer, 0, input_xrag['input_ids'].shape[0])
        generated_output = self.model.generate(
                input_ids = input_xrag['input_ids'],
                attention_mask = input_xrag['attention_mask'],
                do_sample=False,
                stopping_criteria=stopping_criteria,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                retrieval_embeds = doc_embeds,
                use_cache=False,
            )
        decoded = self.tokenizer.batch_decode(generated_output, skip_special_tokens=True)
        return decoded



    def collate_fn(self, examples, eval=False, **kwargs):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]

        docs = [ e['doc'][0] for e in examples]
        label = [e['label'] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)

        data_dict = {}
        # for inference just format and tokenize instruction 
        
        data_dict.update({
            'model_input': (docs, instr),
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        })

        return data_dict

    def format_instruction(self, sample):
        question = sample['query']
        rag_template = """[INST] Refer to the background document and answer the questions:

        Background: {document}

        Question: {question} [/INST] The answer is:"""

        return rag_template.format_map(dict(question=question, document=self.XRAG_TOKEN))
