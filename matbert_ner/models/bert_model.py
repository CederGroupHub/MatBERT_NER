import numpy as np
import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers import BertTokenizer
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel
from matbert_ner.models.crf_layer import CRF
from matbert_ner.models.valid_sequence_output import valid_sequence_output


class BERTNER(BertPreTrainedModel):
    '''
    An BERT model with additional layers for a downstream NER task
    '''
    def __init__(self, model_file, classes, scheme, seed=None):
        '''
        Initializes the BERT NER model
            Arguments:
                model_file: Path to the pretrained BERT model
                classes: A list of classes (labels)
                scheme: The labeling scheme e.g. IOB1, IOB2, or IOBES
                seed: Random seed for parameter initialization
            Returns:
                BERTNER model
        '''
        # model file
        self.model_file = model_file
        # generate configuration from file
        self.config = AutoConfig.from_pretrained(self.model_file)
        # initialize tokenizer from file
        self.tokenizer = BertTokenizer.from_pretrained(self.model_file)
        # initialize pretrained BERT model parent class
        super(BERTNER, self).__init__(self.config)
        # classes
        self.classes = classes
        # labeling scheme
        self.scheme = scheme
        # seed for parameter initialization
        self.seed = seed
        # build model layers
        self.build_model()
    

    def build_model(self):
        '''
        Builds BERT NER model layers
            Arguments:
                None
            Returns:
                None
        '''
        # set seeds if a seed was provided
        if self.seed:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)
        # initialize BERT model from file
        self.bert = BertModel(self.config).from_pretrained(self.model_file)
        # dropout layer for bert output
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        # dense classification layer
        self.classifier = nn.Linear(self.config.hidden_size, len(self.classes))
        # CRF output layer
        self.crf = CRF(classes=self.classes, scheme=self.scheme, batch_first=True)
        # initialize CRF with seed
        self.crf.initialize(self.seed)
    

    def forward(self, input_ids, label_ids=None, attention_mask=None, valid_mask=None, return_logits=False, device='cpu'):
        '''
        BERT NER forward call function
            Arguments:
                input_ids: Batch of sequence ids
                label_ids: Batch of label ids
                attention_mask: Batch of attention masks
                valid_mask: Batch of valid masks
                return_logits: Boolean controlling whether logits are returned
                device: Device used for computation
            Returns:
                always returns prediction_ids
                additionally returns loss if label_ids are provided
                additionally returns logits if specified
                order: loss, logits, prediction_ids
        '''
        # BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=None, position_ids=None,
                            head_mask=None, inputs_embeds=None,
                            output_hidden_states=False)
        # final hidden layer
        sequence_output = outputs[0]
        # valid outputs
        sequence_output, label_ids, attention_mask = valid_sequence_output(sequence_output, label_ids, attention_mask, valid_mask, device)
        # dropout on valid hidden layer output
        sequence_output = self.dropout(sequence_output)
        # classification logits
        logits = self.classifier(sequence_output)
        # prediction ids from Viterbi decode
        prediction_ids = self.crf.decode(logits, mask=attention_mask)
        # if labels are provided, calculate loss
        if label_ids is not None:
            label_ids = label_ids.type(torch.long)
            loss = -self.crf(logits, label_ids, mask=attention_mask)
        # return statements
        if return_logits and label_ids is not None:
            return loss, logits, prediction_ids
        elif label_ids is not None:
            return loss, prediction_ids
        elif return_logits:
            return logits, prediction_ids
        else:
            return prediction_ids
