import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from models.crf_layer import CRF
from models.valid_sequence_output import valid_sequence_output


class BERTNER(BertPreTrainedModel):
    def __init__(self, model_file, classes, scheme, seed):
        self.model_file = model_file
        self.config = AutoConfig.from_pretrained(self.model_file)
        self.tokenizer = BertTokenizer.from_pretrained(self.model_file)
        super(BERTNER, self).__init__(self.config)
        self.classes = classes
        self.scheme = scheme
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
        self.crf = CRF(classes=self.classes, scheme=self.scheme, batch_first=True)
        self.crf.initialize(self.seed)
    

    def forward(self, input_ids, label_ids=None, attention_mask=None, valid_mask=None, return_logits=False, device='cpu'):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=None, position_ids=None,
                            head_mask=None, inputs_embeds=None,
                            output_hidden_states=False)
        sequence_output = outputs[0]
        sequence_output, label_ids, attention_mask = valid_sequence_output(sequence_output, label_ids, attention_mask, valid_mask, device)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        prediction_ids = self.crf.decode(logits, mask=attention_mask)
        if label_ids is not None:
            loss = -self.crf(logits, label_ids, mask=attention_mask)
        if return_logits and label_ids is not None:
            return loss, logits, prediction_ids
        elif label_ids is not None:
            return loss, prediction_ids
        elif return_logits:
            return logits, prediction_ids
        else:
            return prediction_ids
