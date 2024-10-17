'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''

from transformers import Trainer
import torch
from torch.utils.data import DataLoader
from transformers.integrations import WandbCallback
import copy 

# custom trainer
class RAGTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        model_prediction_step = kwargs.pop('model_prediction_step')
        model_generate = kwargs.pop('generate')
        call_back_data = kwargs.pop('call_back_data')
        super().__init__(*args, **kwargs)
        self.model_prediction_step = model_prediction_step
        if self.args.report_to =="wandb":
            #write sample  generation to wandb
            progress_callback = WandbPredictionProgressCallback(
                trainer=self,
                tokenizer=self.tokenizer,
                generation_step= model_generate,
                dataloader=call_back_data,
            )
            # Add the callback to the trainer
            self.add_callback(progress_callback)
    def compute_loss(self, model, inputs):
        model_input, label_ids = inputs['model_input'], inputs['label_ids']
        _, loss = self.model_prediction_step(model, model_input, label_ids=label_ids)
        return loss
    
    def prediction_step(self, model, inputs, prediction_loss_only=None, ignore_keys=None, **kwargs):
        with torch.no_grad():
            #print("prediction step")
            model_input, label_ids = inputs['model_input'], inputs['label_ids']
            #print(model_input)
            #print(label_ids)
            logits, loss = self.model_prediction_step(model, model_input=model_input, label_ids=label_ids)
            #print(loss)
            return (loss, logits, label_ids)
    
class WandbPredictionProgressCallback(WandbCallback):
    # This callback calls the generation function and logs some generated examples to a table in wandb
    # to monitor generation quality while training. The callback is triggered after each evaluation
    def __init__(self, trainer, tokenizer, generation_step, dataloader):
        super().__init__()
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        columns = ["epoch", "predictions"]
        self.wandb_table = self._wandb.Table(columns=columns)
        self.generation_step = generation_step

    def on_evaluate(self, args, state, control, **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        with torch.no_grad():
            for batch in self.dataloader:
                model_input = batch['model_input']
                generated_responses = self.generation_step(model_input)
                for generated_response in generated_responses:
                    self.wandb_table.add_data(state.epoch, generated_response)

        new_table = self._wandb.Table(
            columns=self.wandb_table.columns, data=self.wandb_table.data
        )
        self._wandb.log({"sample_predictions": new_table})


