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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LambdaLR
from torchtools.optim import RangerLars
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import AdamW
from models.crf_layer import CRF
from models.valid_sequence_output import valid_sequence_output


class BertCRFNERModel(NERModel):


    def initialize_model(self):
        ner_model = BertCrfForNer(self.config, self.classes, self.tag_scheme, self.device).to(self.device)
        return ner_model


    def create_optimizer(self, name):
        if name == 'adamw':
            optimizer = AdamW([{'params': self.model.bert.embeddings.parameters(), 'lr': self.elr},
                               {'params': self.model.bert.encoder.parameters(), 'lr': self.tlr},
                               {'params': self.model.bert.pooler.parameters(), 'lr': self.clr},
                               {'params': self.model.classifier.parameters(), 'lr': self.clr},
                               {'params': self.model.crf.parameters(), 'lr': self.clr}])
        if name == 'rangerlars':
            optimizer = RangerLars([{'params': self.model.bert.embeddings.parameters(), 'lr': self.elr},
                                    {'params': self.model.bert.encoder.parameters(), 'lr': self.tlr},
                                    {'params': self.model.bert.pooler.parameters(), 'lr': self.clr},
                                    {'params': self.model.classifier.parameters(), 'lr': self.clr},
                                    {'params': self.model.crf.parameters(), 'lr': self.clr}])
        return optimizer


    def create_scheduler(self, optimizer, n_epochs):
        linear = lambda epoch: (n_epochs-epoch)/(n_epochs)
        # scheduler = CosineAnnealingLR(optimizer=optimizer, T_max=n_epochs, eta_min=0.0, last_epoch=-1, verbose=True)
        scheduler = LambdaLR(optimizer, lr_lambda=linear)
        return scheduler


    def document_embeddings(self, **inputs):
        return self.model.document_embedding(**inputs)


class BertCrfForNer(BertPreTrainedModel):
    def __init__(self, config, tag_names, tag_scheme, device):
        super(BertCrfForNer, self).__init__(config)
        self.bert = BertModel(config).from_pretrained(config.model_name)
        self._device = device
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(tag_names=tag_names, tag_scheme=tag_scheme, batch_first=True)
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
                labels=None, return_logits=False):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                            head_mask=head_mask, inputs_embeds=inputs_embeds,
                            output_hidden_states=False)
        # sequence_output = [outputs[2][i] for i in (-1, -2, -3, -4)]
        # sequence_output = torch.mean(torch.stack(sequence_output), dim=0)
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, self.device)
        sequence_output = self.dropout(sequence_output)
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
