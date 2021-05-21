import torch
from torch import nn
import torchcrf
import numpy as np

class CRF(nn.Module):
    def __init__(self, classes, scheme, batch_first):
        super().__init__()
        # classes
        self.classes = classes
        # class prefixes
        self.prefixes = set([_class.split('-')[0] for _class in self.classes])
        # labeling scheme
        self.scheme = scheme
        # initialize CRF
        self.crf = torchcrf.CRF(num_tags=len(self.classes), batch_first=batch_first)
    

    def initialize(self, seed):
        # set seeds
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        # initialize weights
        self.crf.reset_parameters()
        # construct definitions of invalid transitions
        self.define_invalid_crf_transitions()
        # initialize transitions
        self.init_crf_transitions()
    

    def define_invalid_crf_transitions(self):
        ''' function for establishing valid tagging transitions, assumes BIO or BILUO tagging '''
        if self.scheme == 'IOB':
            # (B)eginning (I)nside (O)utside
            # must begin with O (outside) due to [CLS] token
            self.invalid_begin = ('B', 'I')
            # must end with O (outside) due to [SEP] token
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to B (beginning) or O (outside) - B must be followed by I
            self.invalid_transitions_position = {'B': 'BO'}
            # prevent B (beginning) going to I (inside) or B (beginning) of a different type
            self.invalid_transitions_tags = {'B': 'IB'}
        elif self.scheme == 'IOB2':
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
        elif self.scheme == 'IOBES':
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
    

    def init_crf_transitions(self, imp_value=-10000):
        num_tags = len(self.classes)
        # penalize bad beginnings and endings
        for i in range(num_tags):
            _class = self.classes[i]
            if _class.split('-')[0] in self.invalid_begin:
                torch.nn.init.constant_(self.crf.start_transitions[i], imp_value)
            if _class.split('-')[0] in self.invalid_end:
                torch.nn.init.constant_(self.crf.end_transitions[i], imp_value)
        # build label type dictionary
        label_is = {}
        for label_position in self.prefixes:
            label_is[label_position] = [i for i, _class in enumerate(self.classes) if _class.split('-')[0] == label_position]
        # penalties for invalid consecutive labels by position
        for from_label, to_label_list in self.invalid_transitions_position.items():
            to_labels = list(to_label_list)
            for from_label_i in label_is[from_label]:
                for to_label in to_labels:
                    for to_label_i in label_is[to_label]:
                        torch.nn.init.constant_(self.crf.transitions[from_label_i, to_label_i], imp_value)
        # penalties for invalid consecutive labels by label
        for from_label, to_label_list in self.invalid_transitions_tags.items():
            to_labels = list(to_label_list)
            for from_label_i in label_is[from_label]:
                for to_label in to_labels:
                    for to_label_i in label_is[to_label]:
                        if self.classes[from_label_i].split('-')[1] != self.classes[to_label_i].split('-')[1]:
                            torch.nn.init.constant_(self.crf.transitions[from_label_i, to_label_i], imp_value)
    

    def decode(self, emissions, mask):
        # verterbi decode logits (emissions) using valid attention mask
        crf_out = self.crf.decode(emissions, mask=mask)
        return crf_out


    def forward(self, emissions, labels, mask, reduction='token_mean'):
        # calculate loss with forward pass of crf given logits (emissions) and valid attention mask
        # loss is mean over tokens
        crf_loss = self.crf(emissions, tags=labels, mask=mask, reduction=reduction)
        return crf_loss