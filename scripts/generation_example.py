from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'

def format_instruction(sample):
    return f"""Is the candidate answer semantically or lexically equivalent to the reference answer regarding the question? The candidate should contain at least the same (or more) relevant information as the reference but should not omit any relevant information present in the reference. Output {{equivalent}} or {{not equivalent}}.
    Question: {sample['question']}
    Reference: {sample['reference']}
    Candidate: {sample['candidate']}
    Output: {{"""


quant_config = BitsAndBytesConfig(
load_in_4bit=True,
bnb_4bit_quant_type='nf4',
bnb_4bit_compute_dtype='bfloat16',
bnb_4bit_use_dobule_quant=False
)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', quantization_config=quant_config, attn_implementation="flash_attention_2")
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
tokenizer.pad_token = tokenizer.bos_token
model.config.use_cache = False
model.eval()
pos_tokenid, neg_tokenid = tokenizer.encode('\nequivalent', add_special_tokens=False)[-2], tokenizer.encode('\nnot equivalent', add_special_tokens=False)[-2]



example = {'question': 'this is the question.', 'reference': '44 thousend motors.', 'candidate': 'this is the candidate.'}
instr = format_instruction(example)
instr_tokenized = tokenizer(instr, return_tensors='pt')


print(pos_tokenid, neg_tokenid)
scores = model.generate(**instr_tokenized.to('cuda'), max_new_tokens=1, do_sample=False, output_scores=True, return_dict_in_generate=True).scores
scores = torch.stack(scores)
scores = scores[0, :, [neg_tokenid, pos_tokenid]].float()
pos_prob = torch.softmax(scores, 1)[:, 1].item()
print(pos_prob)
