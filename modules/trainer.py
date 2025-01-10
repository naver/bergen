'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import torch
import numpy as np
from transformers import Trainer
from collections import defaultdict
from torch.utils.data import DataLoader
from utils import evaluate_retrieval_simple

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

class JointTrainer(Trainer):
    def __init__(self, *args, custom_test_dataset=None, qrel=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_test_dataset = custom_test_dataset
        self.qrel = qrel
        for n, p in self.model.named_parameters():
            print(n, p.requires_grad)
        self.additional_losses = {"ranking_loss": list(), "gen_loss": list()}
        self.num_eval_samples = 0

    def evaluate_ranking(self, custom_test_dataset):
        test_dataloader = DataLoader(custom_test_dataset,
                                     collate_fn=self.model.collate_fn_rerank,
                                     batch_size=72,
                                     shuffle=False,
                                     num_workers=0,
                                     )  
        self.model.eval()
        run = defaultdict(dict)
        with torch.no_grad():
            for batch in test_dataloader:
                for k, v in batch.items():
                    if k not in ["q_id", "d_id"]:
                        batch[k] = v.to(self.model.device)
                _, ranking_scores, _ = self.model.compress(**{k: v for k, v in batch.items() if k not in {"q_id", "d_id"}})
                for q_id, d_id, s in zip(
                    batch["q_id"], batch["d_id"], ranking_scores.detach().cpu().tolist()
                ):
                    run[str(q_id)][str(d_id)] = s
        metrics = evaluate_retrieval_simple(run, self.qrel, {"ndcg_cut"})
        ndcg_10 = sum([d["ndcg_cut_10"] for d in metrics.values()]) / len(metrics)
        ndcg_30 = sum([d["ndcg_cut_30"] for d in metrics.values()]) / len(metrics)
        ndcg_100 = sum([d["ndcg_cut_100"] for d in metrics.values()]) / len(metrics)
        return {"ndcg_10": ndcg_10, "ndcg_30": ndcg_30, "ndcg_100": ndcg_100}


    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Overrides the default evaluation to include custom test set evaluation at each eval step
        """
        # Default evaluation on the provided eval_dataset
        eval_metrics = super().evaluate(eval_dataset=eval_dataset,
                                        ignore_keys=ignore_keys, 
                                        metric_key_prefix=metric_key_prefix)

        # Evaluate on the custom test dataset if provided
        if self.custom_test_dataset is not None:
            print("Evaluating on the custom test set...")
            test_metrics = self.evaluate_ranking(self.custom_test_dataset)
            for k, v in test_metrics.items():
                eval_metrics[f"{metric_key_prefix}_{k}"] = v
        # adding custom losses also
        eval_metrics[f"{metric_key_prefix}_ranking_loss"] = np.mean(self.additional_losses["ranking_loss"])
        eval_metrics[f"{metric_key_prefix}_gen_loss"] = np.mean(self.additional_losses["gen_loss"])
        # re-init losses:
        self.additional_losses["ranking_loss"] = list()
        self.additional_losses["gen_loss"] = list()
        self.log(eval_metrics)
        return eval_metrics
    
    def forward_step(self, model, inputs):
        enc_input_ids = inputs['enc_input_ids']
        enc_attention_mask = inputs['enc_attention_mask']
        dec_input_ids = inputs['dec_input_ids']
        dec_attention_mask = inputs['dec_attention_mask']
        label_ids = inputs['labels']
        rr_scores = inputs['rr_scores']
        
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
        rr_scores = rr_scores.to(device)

        # Perform the forward pass
        output = model.forward(
            enc_input_ids=enc_input_ids,
            enc_attention_mask=enc_attention_mask,
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_attention_mask,
            label=label_ids,
            rr_scores=rr_scores,
        )
        return output

    def compute_loss(self, model, inputs, return_outputs=False):
        output = self.forward_step(model, inputs)
        return output['total_loss']
    
    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None, **kwargs):
        
        with torch.no_grad():
            output = self.forward_step(model, inputs)
        # it is not direct to record other losses than the "main" one
        # see for instance ==> https://github.com/zipzou/hf-multitask-trainer    
        self.additional_losses["ranking_loss"].append(output['ranking_loss'].cpu().detach().item())
        self.additional_losses["gen_loss"].append(output['loss'].cpu().detach().item())
            
        #return {"total_loss": output['total_loss'], "ranking_loss": output['ranking_loss'], "loss": output["loss"]}, output["logits"], inputs["labels"]
        return output['total_loss'], output["logits"], inputs["labels"]