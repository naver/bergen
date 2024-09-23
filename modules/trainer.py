'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import torch
import numpy as np
from transformers import Trainer


# def compute_metrics(eval_pred, model):
#     """
#     Computes the evaluation metrics, to be used during training
#     """
#     logits, labels = eval_pred
#     if isinstance(logits, tuple):  # Check if logits are wrapped in a tuple
#         logits = logits[0]  # Adjust this to access the correct tuple element
    
#     preds = np.argmax(logits, axis=-1)
#     original_model = model.module if hasattr(model, 'module') else model
#     ignore_positions = labels == -100

#     labels[ignore_positions] = original_model.decoder_tokenizer.pad_token_id
#     preds[ignore_positions] = original_model.decoder_tokenizer.pad_token_id

#     preds_str = original_model.decoder_tokenizer.batch_decode(preds, skip_special_tokens=True)
#     labels_str = original_model.decoder_tokenizer.batch_decode(labels, skip_special_tokens=True)

#     print('_'*15)
#     # print examples
#     for i, (pre, lab) in enumerate(zip(preds_str[:2], labels_str[:2])):
#         print('\n')
#         print('LABEL: ', lab)
#         print('PRED:  ', pre)
        
#     print('_'*15)  
    
#     # reformattting because bergen uses list of labels:
#     labels_str = [[elt] for elt in labels_str] 
#     metrics = RAGMetrics.compute(predictions=preds_str, references=labels_str)
#     return metrics


# custom trainer
class RAGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        #self.model_prediction_step = kwargs.pop('model_prediction_step')
        super().__init__(*args, **kwargs)
        
    def make_prediction_step(self, model, inputs):
        model_input, label_ids = inputs['model_input'], inputs['label_ids']
        output = model.forward(
            enc_input_ids=model_input['enc_input_ids'],
            enc_attention_mask=model_input['enc_attention_mask'],
            dec_input_ids=model_input['dec_input_ids'],
            dec_attention_mask=model_input['dec_attention_mask'],
            labels=label_ids
        )
        return output['loss'], output['logits']

    def compute_loss(self, model, inputs):
        #_, loss = self.model_prediction_step(model, model_input, label_ids=label_ids)
        return self.make_prediction_step(model, inputs)[0]
    
    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None, **kwargs):
        with torch.no_grad():
            loss, logits = self.make_prediction_step(model, inputs)
            
        return loss, logits, inputs['label_ids']
