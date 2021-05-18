import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from models.crf_layer import CRF
from models.valid_sequence_output import valid_sequence_output


class BERTNER(BertPreTrainedModel):
    def __init__(self, model_file, tag_names, tag_scheme, seed):
        self.model_file = model_file
        self.config = AutoConfig.from_pretrained(self.model_file)
        super(BERTNER, self).__init__(self.config)
        self.classes = tag_names
        self.tag_scheme = tag_scheme
        self.seed = seed
        self.build_model()
    

    def build_model(self):
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
        self.bert = BertModel(self.config).from_pretrained(self.model_file)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.config.hidden_size, len(self.classes))
        self.crf = CRF(tag_names=self.classes, tag_scheme=self.tag_scheme, batch_first=True)
        self.crf.initialize(self.seed)
    

    def forward(self, input_ids,
                attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None,
                inputs_embeds=None, valid_mask=None,
                labels=None, return_logits=False, device='cpu'):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                            head_mask=head_mask, inputs_embeds=inputs_embeds,
                            output_hidden_states=False)
        sequence_output = outputs[0]
        sequence_output, attention_mask = valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, device)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        predictions = self.crf.decode(logits, mask=attention_mask)
        outputs = (predictions,)
        if return_logits:
            outputs = outputs+(logits,)
        if labels is not None:
            labels = torch.where(labels>=0, labels, torch.zeros_like(labels))
            loss = -self.crf(logits, labels, mask=attention_mask)
            outputs = (loss,)+outputs
        return outputs
