'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

import torch
from torch.nn import functional as F
import tensorflow_hub as hub
from transformers import BertTokenizer
import tensorflow as tf
from tqdm import tqdm

import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.get_logger().setLevel(tf.compat.v1.logging.ERROR)

class BEM:
    def __init__(self, batch_size=2048):
        self.batch_size = batch_size
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


    def bertify_example(self, question, reference, candidate, max_length=512):
        question = self.tokenizer.tokenize(question)[:max_length]
        reference = self.tokenizer.tokenize(reference)[:max_length]
        candidate = self.tokenizer.tokenize(candidate)[:max_length]
        
        tokens = ['[CLS]'] + candidate + ['[SEP]'] + reference + ['[SEP]'] + question + ['[SEP]']
        
        input_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(tokens))
        segment_ids = torch.tensor([0] * (len(candidate) + 2) + [1] * (len(reference) + 1) + [2] * (len(question) + 1))

        input_ids = F.pad(torch.tensor(input_ids), (0, max_length - len(input_ids)), value=0)
        segment_ids = F.pad(torch.tensor(segment_ids), (0, max_length - len(segment_ids)), value=0)
        
        return {'input_ids': input_ids, 'segment_ids': segment_ids}


    def bertify_examples(self, examples, max_length=512):
        input_ids = []
        segment_ids = []
        for example in examples:
            question = example['question']
            candidate = example['candidate']
            reference = example['reference']

            if isinstance(reference, str):
                reference = [reference]

            for ref in reference:
                example_inputs = self.bertify_example(question, ref, candidate, max_length=max_length)

            input_ids.append(example_inputs['input_ids'])
            segment_ids.append(example_inputs['segment_ids'])

        return {'input_ids': torch.stack(input_ids), 'segment_ids': torch.stack(segment_ids)}

    def __call__(self, predictions, references, questions):
        # Loading the TensorFlow Hub model
        self.model = hub.load('https://tfhub.dev/google/answer_equivalence/bem/1')
        assert len(predictions) == len(references) == len(questions)
        examples = [{'question': questions[i], 'reference': references[i], 'candidate': predictions[i]}  for i in range(len(predictions))]
        
        inputs = self.bertify_examples(examples, max_length=self.tokenizer.model_max_length)
        # The outputs are raw logits.
        scores = list()
        # Perform batch inference
        for i in tqdm(range(0, len(inputs['input_ids']), self.batch_size), desc='BEM evaluation...'):
            # Extract batch
            batch_input_ids = inputs['input_ids'][i:i+self.batch_size]
            batch_segment_ids = inputs['segment_ids'][i:i+self.batch_size]
            inp = {"input_ids": tf.stop_gradient(batch_input_ids), "segment_ids": tf.stop_gradient(batch_segment_ids)}
            raw_outputs = self.model(inp)
            raw_outputs_torch = torch.from_numpy(raw_outputs.numpy())
            scores.append(raw_outputs_torch)
        # They can be transformed into a classification 'probability' like so:
        del self.model
        scores = torch.cat(scores)
        tf.keras.backend.clear_session()
        torch.cuda.empty_cache()
        scores = F.softmax(scores, dim=1)[:, 1]
        return scores.mean().item(), scores


