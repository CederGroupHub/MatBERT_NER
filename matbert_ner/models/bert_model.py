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
import torchcrf


class BertCRFNERModel(NERModel):


    def initialize_model(self):
        ner_model = BertCrfForNer(self.config, self.classes, self.tag_format, self.device).to(self.device)
        return ner_model


    def create_optimizer(self, full_finetuning=True):
        if full_finetuning:
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
                labels=None):
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
        # sequence_output = [outputs[2][i] for i in (-1, -2, -3, -4)]
        # sequence_output = torch.mean(torch.mean(torch.stack(sequence_output), dim=0), dim=1)
        sequence_output = torch.mean(outputs[0], dim=1)
        return sequence_output


def valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, device):
    batch_size, max_len, feat_dim = sequence_output.shape
    valid_output = torch.zeros(batch_size, max_len, feat_dim, dtype=torch.float32, device=device)
    valid_attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        jj = -1
        for j in range(max_len):
            if valid_mask[i][j].item() == 1:
                jj += 1
                valid_output[i][jj] = sequence_output[i][j]
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask


class CRF(nn.Module):
    def __init__(self, tag_names, tag_format, batch_first):
        super().__init__()
        # tag names
        self.tag_names = tag_names
        # tag prefixes
        self.prefixes = set([tag_name.split('-')[0] for tag_name in self.tag_names])
        # tag format
        self.tag_format = tag_format
        # initialize CRF
        self.crf = torchcrf.CRF(num_tags=len(self.tag_names), batch_first=batch_first)
    

    def initialize(self):
        # initialize weights
        self.crf.reset_parameters()
        # construct definitions of invalid transitions
        self.define_invalid_crf_transitions()
        # initialize transitions
        self.init_crf_transitions()
    

    def define_invalid_crf_transitions(self):
        ''' function for establishing valid tagging transitions, assumes BIO or BILUO tagging '''
        if self.tag_format == 'IOB':
            # (B)eginning (I)nside (O)utside
            # must begin with O (outside) due to [CLS] token
            self.invalid_begin = ('B', 'I')
            # must end with O (outside) due to [SEP] token
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to B (beginning) or O (outside) - B must be followed by I
            self.invalid_transitions_position = {'B': 'BO'}
            # prevent B (beginning) going to I (inside) or B (beginning) of a different type
            self.invalid_transitions_tags = {'B': 'IB'}
        elif self.tag_format == 'IOB2':
            # (B)eginning (I)nside (O)utside
            # must begin with O (outside) due to [CLS] token
            self.invalid_begin = ('B', 'I')
            # must end with O (outside) due to [SEP] token
            self.invalid_end = ('B', 'I')
            # prevent O (outside) going to I (inside) - O must be followed by B or O
            self.invalid_transitions_position = {'O': 'I'}
            # prevent B (beginning) going to I (inside) of a different type
            # prevent I (inside) going to I (inside) of a different type
            self.invalid_transitions_tags = {'B': 'I',
                                             'I': 'I'}
        elif self.tag_format == 'IOBES':
            # (B)eginning (I)nside (E)nd (S)ingle (O)utside
            # must begin with O (outside) due to [CLS] token
            self.invalid_begin = ('B', 'I', 'E', 'S')
            # must end with O (outside) due to [SEP] token
            self.invalid_end = ('B', 'I', 'E', 'S')
            # prevent B (beginning) going to B (beginning), O (outside), or S (single) - B must be followed by I or E
            # prevent I (inside) going to B (beginning), O (outside), or S (single) - I must be followed by I or E
            # prevent E (end) going to I (inside) or E (end) - U must be followed by B, O, or U
            # prevent S (single) going to I (inside) or E (end) - U must be followed by B, O, or U
            # prevent O (outside) going to I (inside) or E (end) - O must be followed by B, O, or U
            self.invalid_transitions_position = {'B': 'BOS',
                                                 'I': 'BOS',
                                                 'E': 'IE',
                                                 'S': 'IE',
                                                 'O': 'IE'}
            # prevent B (beginning) from going to I (inside) or E (end) of a different type
            # prevent I (inside) from going to I (inside) or E (end) of a different tpye
            self.invalid_transitions_tags = {'B': 'IE',
                                             'I': 'IE'}
    

    def init_crf_transitions(self, imp_value=-100):
        num_tags = len(self.tag_names)
        # penalize bad beginnings and endings
        for i in range(num_tags):
            tag_name = self.tag_names[i]
            if tag_name.split('-')[0] in self.invalid_begin:
                torch.nn.init.constant_(self.crf.start_transitions[i], imp_value)
            if tag_name.split('-')[0] in self.invalid_end:
                torch.nn.init.constant_(self.crf.end_transitions[i], imp_value)
        # build tag type dictionary
        tag_is = {}
        for tag_position in self.prefixes:
            tag_is[tag_position] = [i for i, tag in enumerate(self.tag_names) if tag.split('-')[0] == tag_position]
        # penalties for invalid consecutive tags by position
        for from_tag, to_tag_list in self.invalid_transitions_position.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        torch.nn.init.constant_(self.crf.transitions[from_tag_i, to_tag_i], imp_value)
        # penalties for invalid consecutive tags by tag
        for from_tag, to_tag_list in self.invalid_transitions_tags.items():
            to_tags = list(to_tag_list)
            for from_tag_i in tag_is[from_tag]:
                for to_tag in to_tags:
                    for to_tag_i in tag_is[to_tag]:
                        if self.tag_names[from_tag_i].split('-')[1] != self.tag_names[to_tag_i].split('-')[1]:
                            torch.nn.init.constant_(self.crf.transitions[from_tag_i, to_tag_i], imp_value)
    

    def decode(self, emissions, mask):
        crf_out = self.crf.decode(emissions, mask=mask)
        return crf_out


    def forward(self, emissions, tags, mask, reduction='mean'):
        crf_loss = self.crf(emissions, tags=tags, mask=mask, reduction=reduction)
        return crf_loss
