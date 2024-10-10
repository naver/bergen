'''
BERGEN
Copyright (c) 2024-present NAVER Corp.
CC BY-NC-SA 4.0 license
'''
import torch
from transformers.integrations import WandbCallback

    
class WandbPredictionProgressCallback(WandbCallback):
    # This callback calls the generation function and logs some generated examples to a table in wandb
    # to monitor generation quality while training. The callback is triggered after each evaluation
    def __init__(self, tokenizer, generation_step, dataloader):
        super().__init__()
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

        # TODO: we could easily add bergen metrics here

        new_table = self._wandb.Table(
            columns=self.wandb_table.columns, data=self.wandb_table.data
        )
        self._wandb.log({"sample_predictions": new_table})


