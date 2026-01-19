import torch
from torch import nn
from dataclasses import dataclass
from typing import Optional, Union, Tuple
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel
from transformers.modeling_outputs import ModelOutput
from torch.nn import CrossEntropyLoss, MSELoss

@dataclass
class RankingCompressionOutput(ModelOutput):
    """ """

    loss: Optional[torch.FloatTensor] = None
    compression_loss: Optional[torch.FloatTensor] = None
    ranking_loss: Optional[torch.FloatTensor] = None
    compression_logits: torch.FloatTensor = None
    ranking_scores: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class XLMRobertaForCompressionAndRanking(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        num_labels = getattr(config, "num_labels", 2)
        self.num_labels = num_labels
        self.roberta = XLMRobertaModel(config)
        output_dim = config.hidden_size

        ### RANKING LAYER
        self.classifier = nn.Linear(output_dim, num_labels)
        drop_out = getattr(config, "cls_dropout", None)
        drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
        self.dropout = nn.Dropout(drop_out)

        ### COMPRESSION LAYER: another head (initialized randomly)
        token_dropout = drop_out
        self.token_dropout = nn.Dropout(token_dropout)
        self.token_classifier = nn.Linear(
            config.hidden_size, 2
        )  # => hard coded number of labels

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        ranking_labels: Optional[torch.LongTensor] = None,
        loss_weight: Optional[float] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], RankingCompressionOutput]:
        """simplified forward"""
        outputs = self.roberta(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoder_layer = outputs[0]
        pooled_output = outputs['pooler_output']
        pooled_output = self.dropout(pooled_output)
        ranking_logits = self.classifier(pooled_output)
        compression_logits = self.token_classifier(self.token_dropout(encoder_layer))
        ranking_scores = ranking_logits[:, 0].squeeze()  # select first dim of logits for ranking scores

        compression_loss = None
        ranking_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(compression_logits.device)
            loss_fct = CrossEntropyLoss()
            compression_loss = loss_fct(compression_logits.view(-1, 2), labels.view(-1))
        if ranking_labels is not None:
            # here ranking labels are scores (from a teacher) we aim to directly distil (pointwise MSE)
            ranking_labels = ranking_labels.to(ranking_logits.device)
            loss_fct = MSELoss()
            ranking_loss = loss_fct(ranking_scores, ranking_labels.squeeze())
        loss = None
        if (labels is not None) and (ranking_labels is not None):
            w = loss_weight if loss_weight else 1
            loss = compression_loss + w * ranking_loss
        elif labels is not None:
            loss = compression_loss
        elif ranking_labels is not None:
            loss = ranking_loss

        return RankingCompressionOutput(
            loss=loss,
            compression_loss=compression_loss,
            ranking_loss=ranking_loss,
            compression_logits=compression_logits,
            ranking_scores=ranking_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )