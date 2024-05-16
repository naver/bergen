'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''


from models.generators.generator import Generator
class RandomAnswer(Generator):
    # this class serves as a baseline for evaluation
    # it will output a label of a random query
    def __init__(self, 
                 model_name=None, 
                 **kwargs
                 ):
        self.model_name = model_name

    def tokenizer(self, instr, **kwargs):
        return instr

    def format_instruction(self, sample):
        # return the reference[0] of a random example, as we saved those in the ranking_labels of the queries
        return f"""{sample['ranking_label']}"""
    
    def generate(self, inp):
        return inp

    def collate_fn(self, examples, eval=False):
        q_ids = [e['q_id'] for e in examples]
        instr = [self.format_instruction(e) for e in examples]
        label = [[e['label']] if isinstance(e['label'], str) else e['label'] for e in examples]
        query = [e['query'] for e in examples]
        ranking_label = [e['ranking_label'] for e in examples] if 'ranking_label' in examples[0] else [None] * len(examples)
        return {
            'model_input': instr,
            'q_id': q_ids, 
            'query': query, 
            'instruction': instr,
            'label': label, 
            'ranking_label': ranking_label,
        }
        