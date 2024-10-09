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

# BASE DEEPSPEED CONFIG STAGE 3
ds_config = {
        "fp16": {
            "enabled": 'auto'  # Enable mixed precision for speedup and memory efficiency
        },
        "bf16": {
            "enabled": 'auto'
        }, 
        "logging": {
            "level": "debug",  # Options: debug, info, warning, error, critical
            "file_path": "/scratch/1/user/mlouis/calmar/deepspeed.log",  # Path to store the logs
            "log_level_steps": 1  # Log after every 50 steps
        },
        "zero_optimization": {
            "stage": 3,  # Enable ZeRO Stage 3 for full memory optimization (optimizer, gradients, and parameters)
            
            # # Offload optimizer states to CPU (optional: can offload to NVMe for better performance)
            "offload_optimizer": {
                "device": "cpu",  # Offload optimizer states to CPU
                "pin_memory": True
            },

            # # Offload parameters to CPU to free GPU memory (optional: can offload to NVMe for better performance)
            "offload_param": {
                "device": "cpu",  # Offload model parameters to CPU
                "pin_memory": True
            },

            # Other memory optimization parameters for ZeRO Stage 3
            "allgather_partitions": True,
            "reduce_scatter": True,
            "contiguous_gradients": True,
            # These options are specific to ZeRO Stage 3
            "stage3_prefetch_bucket_size": 1e7,  # Controls the size of the bucket used to prefetch parameters from CPU to GPU: larger is faster but greedier
            "stage3_max_live_parameters": 1e10,   # Limit for live parameters in memory to avoid memory overflow
            "stage3_param_persistence_threshold": 1e6  # Threshold for partitioning parameters, larger ones stay on GPU
        },
        # "zero_optimization": {
        #     "stage": 2,  # Enable ZeRO optimization for memory efficiency
        #     "offload_optimizer": {
        #         "device": "cpu",  # Offload optimizer states to CPU
        #         "pin_memory": True
        #     },
        #     "allgather_partitions": True,
        #     "reduce_scatter": True,
        #     "contiguous_gradients": True
        # },
        "train_batch_size": 'auto',  # Total batch size across all GPUs and accumulation steps
        "train_micro_batch_size_per_gpu": 'auto',  # Batch size per GPU
        "gradient_accumulation_steps": 'auto',  # Number of steps to accumulate gradients
        "gradient_clipping": 'auto',  # Clip gradients to avoid exploding gradients
        "steps_per_print": 'auto'  # How often to print statistics
    }


# custom trainer
class RAGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        #self.model_prediction_step = kwargs.pop('model_prediction_step')
        super().__init__(*args, **kwargs)
        
    def make_prediction_step(self, model, inputs):
        model_input = inputs['model_input']
        
        # Ensure inputs are on the same device as the model using accelerator.move_to_device
        enc_input_ids = model_input['enc_input_ids']
        enc_attention_mask = model_input['enc_attention_mask']
        dec_input_ids = model_input['dec_input_ids']
        dec_attention_mask = model_input['dec_attention_mask']
        label_ids = inputs['label_ids']
        
            # Check if the model is wrapped in DataParallel or DistributedDataParallel
        if hasattr(model, "module"):
            device = model.module.device  # Get device from the underlying model
        else:
            device = model.device  # Get device if it's not wrapped in DataParallel

        # Move inputs to the correct device of the model
        enc_input_ids = enc_input_ids.to(device)
        enc_attention_mask = enc_attention_mask.to(device)
        dec_input_ids = dec_input_ids.to(device)
        dec_attention_mask = dec_attention_mask.to(device)
        label_ids = label_ids.to(device)

        # Perform the forward pass
        output = model.forward(
            enc_input_ids=enc_input_ids,
            enc_attention_mask=enc_attention_mask,
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_attention_mask,
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
