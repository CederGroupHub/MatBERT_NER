from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.modeling_bert import BertPreTrainedModel
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Optional
import numpy as np
from models.base_ner_model import NERModel
import torch.optim as optim
from torchtools.optim import RangerLars
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from models.crf_layer import CRF
from models.valid_sequence_output import valid_sequence_output


class BertCRFNERModel(NERModel):


    def initialize_model(self):
        ner_model = BertCrfForNer(self.config, self.classes, self.tag_format, self.device).to(self.device)
        return ner_model


    def create_optimizer(self, deep_finetuning=True):
        if deep_finetuning:
            param_optimizer = list(self.model.named_parameters())
            no_decay = ['bias', 'gamma', 'beta']
            optimizer_grouped_parameters = [{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0},
                                            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],'weight_decay_rate': 0.0}]
        else:
            param_optimizer = [item for sblst in [list(module.named_parameters()) for module in self.model.model_modules[1:]] for item in sblst]
            optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr, eps=1e-8)
        # optimizer = RangerLars(optimizer_grouped_parameters, lr=self.lr)
        return optimizer


    def create_scheduler(self, optimizer, n_epochs, train_dataloader):
        warmup_epochs = 1
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=len(train_dataloader)*warmup_epochs,
                                                    num_training_steps=(n_epochs-warmup_epochs)*len(train_dataloader))
        # scheduler = get_cosine_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=len(train_dataloader)*warmup_epochs,
        #                                             num_training_steps=(n_epochs-warmup_epochs)*len(train_dataloader),
        #                                             num_cycles=(n_epochs-warmup_epochs)/10)
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
        #                                                                num_warmup_steps=len(train_dataloader)*warmup_epochs,
        #                                                                num_training_steps=(n_epochs-warmup_epochs)*len(train_dataloader),
        #                                                                num_cycles=(n_epochs-warmup_epochs)/5)
        return scheduler


    def document_embeddings(self, **inputs):
        return self.model.document_embedding(**inputs)


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, tag_names, tag_format, device):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config).from_pretrained(config.model_name)
        self._device = device
        self.use_lstm = False
        self.dropout_b = nn.Dropout(config.hidden_dropout_prob)
        self.model_modules = [self.bert, self.dropout_b]
        if self.use_lstm:
            self.lstm = nn.LSTM(batch_first=True, input_size=config.hidden_size,
                                hidden_size=64, num_layers=2,
                                bidirectional=True, dropout=0.1)
            self.attn = nn.MultiheadAttention(embed_dim=128, num_heads=16, dropout=0.25)
            self.dropout_c = nn.Dropout(0.25)
            self.model_modules.extend([self.lstm, self.attn, self.dropout_c])
        self.classifier = nn.Linear(128 if self.use_lstm else config.hidden_size, config.num_labels)
        self.model_modules.append(self.classifier)
        self.crf = CRF(tag_names=tag_names, tag_format=tag_format, batch_first=True)
        self.crf.initialize()
        self.model_modules.append(self.crf)


    @property
    def device(self):
        return self._device
    

    @device.setter
    def device(self, device):
        self._device = device


    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                inputs_embeds=None, valid_mask=None,
                labels=None, return_logits=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                            head_mask=head_mask, inputs_embeds=inputs_embeds,
                            output_hidden_states=False)
        # sequence_output = [outputs[2][i] for i in (-1, -2, -3, -4)]
        # sequence_output = torch.mean(torch.stack(sequence_output), dim=0)
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, self.device)
        sequence_output = self.dropout_b(sequence_output)
        if self.use_lstm:
            lstm_out, _ = self.lstm(sequence_output)
            attn_out, attn_weight = self.attn(lstm_out, lstm_out, lstm_out, key_padding_mask=attention_mask)
            logits = self.classifier(self.dropout_c(attn_out))
        else:
            logits = self.classifier(sequence_output)
        predictions = self.crf.decode(logits, mask=attention_mask)
        outputs = (predictions, )
        if return_logits:
            outputs = outputs + (logits, )
        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = -self.crf(logits, labels, mask=attention_mask)
            outputs = (loss,) + outputs
        return outputs  # loss, logits/predictions


    def document_embedding(self, input_ids,
                           attention_mask=None, token_type_ids=None,
                           position_ids=None, head_mask=None,
                           inputs_embeds=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                            head_mask=head_mask, inputs_embeds=inputs_embeds,
                            output_hidden_states=True)
        sequence_output = [outputs[2][i] for i in (-1, -2, -3, -4)]
        sequence_output = torch.mean(torch.mean(torch.stack(sequence_output), dim=0), dim=1)
        #sequence_output = torch.mean(outputs[0], dim=1)
        return sequence_output
