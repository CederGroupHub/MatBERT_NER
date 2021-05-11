import torch
from torch import nn
import torchcrf

class CRF(nn.Module):
    def __init__(self, tag_names, tag_scheme, batch_first):
        super().__init__()
        # tag names
        self.tag_names = tag_names
        # tag prefixes
        self.prefixes = set([tag_name.split('-')[0] for tag_name in self.tag_names])
        # tag format
        self.tag_scheme = tag_scheme
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
        if self.tag_scheme == 'IOB':
            # (B)eginning (I)nside (O)utside
            # must begin with O (outside) due to [CLS] token
            self.invalid_begin = ('B', 'I')
            # must end with O (outside) due to [SEP] token
            self.invalid_end = ('B', 'I')
            # prevent B (beginning) going to B (beginning) or O (outside) - B must be followed by I
            self.invalid_transitions_position = {'B': 'BO'}
            # prevent B (beginning) going to I (inside) or B (beginning) of a different type
            self.invalid_transitions_tags = {'B': 'IB'}
        elif self.tag_scheme == 'IOB2':
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
        elif self.tag_scheme == 'IOBES':
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
