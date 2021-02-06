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
import torchcrf

class BertCRFNERModel(NERModel):

    def initialize_model(self):
        ner_model = BertCrfForNer(self.config, self.classes, self.device).to(self.device)
        return ner_model

    def create_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]

        bert_parameters = self.model.bert.named_parameters()
        classifier_parameters = self.model.classifier.named_parameters()
        bert_lr = self.lr
        classifier_lr = self.lr
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": bert_lr},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": bert_lr},

            {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": classifier_lr},
            {"params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": classifier_lr}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, self.lr, eps=1e-8)
        return optimizer

    def create_scheduler(self, optimizer, n_epochs, train_dataloader):
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=len(train_dataloader), num_training_steps=n_epochs*len(train_dataloader), num_cycles=n_epochs/10
        )

        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=0, num_training_steps=n_epochs*len(train_dataloader)
        # )

        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=n_epochs*len(train_dataloader), num_cycles=n_epochs/5
        )

        return scheduler

    def document_embeddings(self, **inputs):
        return self.model.document_embedding(**inputs)


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt) 
        logpt = (1 - pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss

class BertNER(BertPreTrainedModel):
    def __init__(self, config, device):
        super(BertNER, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self._device = device
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device = device

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask, self.device)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = FocalLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                print(logits.view(-1, self.num_labels), labels.view(-1))
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs # (loss), scores, (hidden_states), (attentions)

class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, tag_names, device):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config)
        self._device = device
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(tag_names=tag_names, batch_first=True)
        self.init_weights()
    
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device = device

    def forward(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            valid_mask=None,
            labels=None,
            decode=False,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        sequence_output = [outputs[2][i] for i in (-1, -2, -3, -4)]
        sequence_output = torch.mean(torch.stack(sequence_output), dim=0)
        # sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(sequence_output, valid_mask, attention_mask, self.device)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if decode:
            tags = self.crf.crf.decode(logits, mask=attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf.crf(logits, tags=labels, mask=attention_mask)
            outputs = (-1 * loss,) + outputs

        return outputs  # (loss), scores

    def document_embedding(
            self,
            input_ids,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,

    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True
        )
        sequence_output = [outputs[2][i] for i in (-1, -2, -3, -4)]
        sequence_output = torch.mean(torch.mean(torch.stack(sequence_output), dim=0), dim=1)

        return sequence_output

def valid_sequence_output(sequence_output, valid_mask, attention_mask, device):
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
    def __init__(self, tag_names, batch_first):
        super().__init__()
        # tag pad index and tag names
        self.tag_pad_idx = -100
        self.pad_token = '[PAD]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.tag_names = tag_names
        # initialize CRF
        self.crf = torchcrf.CRF(num_tags=len(self.tag_names), batch_first=batch_first)
        # construct definitions of invalid transitions
        self.define_invalid_crf_transitions()
        # initialize transitions
        self.init_crf_transitions()
    

    def define_invalid_crf_transitions(self):
        ''' function for establishing valid tagging transitions, assumes BIO or BILUO tagging '''
        self.prefixes = set([tag_name[0] for tag_name in self.tag_names if tag_name not in [self.pad_token, self.cls_token, self.sep_token]])
        if self.prefixes == set(['B', 'I', 'O']):
            # (B)eginning (I)nside (O)utside
            # cannot begin sentence with I (inside), only B (beginning) or O (outside)
            self.invalid_begin = ('I',)
            # cannot end sentence with B (beginning) or I (inside) - assumes data ends with O (outside), such as punctuation
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to P - B must be followed by B, I, or O
            # prevent I (inside) going to P - I must be followed by B, I, or O
            # prevent O (outside) going to I (inside) - O must be followed by B or O
            self.invalid_transitions_position = {'B': 'P',
                                                 'I': 'P',
                                                 'O': 'I'}
            # prevent B (beginning) going to I (inside) of a different type
            # prevent I (inside) going to I (inside) of a different type
            self.invalid_transitions_tags = {'B': 'I',
                                             'I': 'I'}
        if self.prefixes == set(['B', 'I', 'L', 'U', 'O']):
            # (B)eginning (I)nside (L)ast (U)nit (O)utside
            # cannot begin sentence with I (inside) or L (last)
            self.invalid_begin = ('I', 'L')
            # cannot end sentence with B (beginning) or I (inside)
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to B (beginning), O (outside), U (unit), or P - B must be followed by I or L
            # prevent I (inside) going to B (beginning), O (outside), U (unit), or P - I must be followed by I or L
            # prevent L (last) going to I (inside) or L(last) - U must be followed by B, O, U, or P
            # prevent U (unit) going to I (inside) or L(last) - U must be followed by B, O, U, or P
            # prevent O (outside) going to I (inside) or L (last) - O must be followed by B, O, U, or P
            self.invalid_transitions_position = {'B': 'BOUP',
                                                 'I': 'BOUP',
                                                 'L': 'IL',
                                                 'U': 'IL',
                                                 'O': 'IL'}
            # prevent B (beginning) from going to I (inside) or L (last) of a different type
            # prevent I (inside) from going to I (inside) or L (last) of a different tpye
            self.invalid_transitions_tags = {'B': 'IL',
                                             'I': 'IL'}
        if self.prefixes == set(['B', 'I', 'E', 'S', 'O']):
            # (B)eginning (I)nside (E)nd (S)ingle (O)utside
            # cannot begin sentence with I (inside) or E (end)
            self.invalid_begin = ('I', 'E')
            # cannot end sentence with B (beginning) or I (inside)
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to B (beginning), O (outside), S (single), or P - B must be followed by I or E
            # prevent I (inside) going to B (beginning), O (outside), S (single), or P - I must be followed by I or E
            # prevent E (end) going to I (inside) or E (end) - U must be followed by B, O, U, or P
            # prevent S (single) going to I (inside) or E (end) - U must be followed by B, O, U, or P
            # prevent O (outside) going to I (inside) or E (end) - O must be followed by B, O, U, or P
            self.invalid_transitions_position = {'B': 'BOSP',
                                                 'I': 'BOSP',
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
            if tag_name[0] in self.invalid_begin or tag_name == self.pad_token:
                torch.nn.init.constant_(self.crf.start_transitions[i], imp_value)
            # don't penalize endings since not every example ends with punctuation
            # if tag_name[0] in self.invalid_end:
            #     torch.nn.init.constant_(self.crf.end_transitions[i], imp_value)
        # build tag type dictionary
        tag_is = {}
        for tag_position in self.prefixes:
            tag_is[tag_position] = [i for i, tag in enumerate(self.tag_names) if tag[0] == tag_position]
        tag_is['P'] = [i for i, tag in enumerate(self.tag_names) if tag == 'tag']
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
