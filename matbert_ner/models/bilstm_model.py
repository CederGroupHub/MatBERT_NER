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
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from models.crf_layer import CRF
from models.valid_sequence_output import valid_sequence_output


class BiLSTMNERModel(NERModel):


    def initialize_model(self):
        ner_model = BiLSTM(self.config, self.classes, self.tag_format, self.device).to(self.device)
        return ner_model


    def create_optimizer(self, full_finetuning = True):
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)
        return optimizer


    def create_scheduler(self, optimizer, n_epochs, train_dataloader):
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=len(train_dataloader),
                                                    num_training_steps=n_epochs*len(train_dataloader),
                                                    num_cycles=n_epochs/10)
        # scheduler = get_linear_schedule_with_warmup(optimizer,
        #                                             num_warmup_steps=0,
        #                                             num_training_steps=n_epochs*len(train_dataloader))
        # scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer,
        #                                                                num_warmup_steps=0,
        #                                                                num_training_steps=n_epochs*len(train_dataloader),
        #                                                                num_cycles=n_epochs/5)
        return scheduler


    def document_embeddings(self, **inputs):
        return self.model.document_embedding(**inputs)


class BiLSTM(nn.Module):
    def __init__(self, config, tag_names, tag_format, device):
        super().__init__()
        self._device = device
        self.lstm_hidden_size = 64
        self.attn_heads = 16
        self.embedding = nn.Embedding(30522, config.hidden_size)
        self.dropout_b = nn.Dropout(config.hidden_dropout_prob)
        self.lstm = nn.LSTM(batch_first=True, input_size=config.hidden_size,
                            hidden_size=self.lstm_hidden_size, num_layers=4,
                            bidirectional=True, dropout=0.1)
        self.attn = nn.MultiheadAttention(embed_dim=self.lstm_hidden_size*2, num_heads=self.attn_heads, dropout=0.25)
        self.dropout_c = nn.Dropout(0.25)
        self.classifier = nn.Linear(2*self.lstm_hidden_size, config.num_labels)
        self.crf = CRF(tag_names=tag_names, tag_format=tag_format, batch_first=True)
        self.crf.initialize()


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
                labels=None):
        sequence_output = self.embedding(input_ids)
        sequence_output, _ = self.lstm(sequence_output)
        sequence_output, attention_mask = valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, self.device)
        sequence_output, sequence_weight = self.attn(sequence_output, sequence_output, sequence_output, key_padding_mask=~attention_mask.permute(1, 0))
        sequence_output = self.dropout_b(sequence_output)
        logits = self.classifier(sequence_output)     
        predictions = self.crf.decode(logits, mask=attention_mask)
        outputs = (predictions,)
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
        return sequence_output
