import torch
from torch import nn
import torchcrf
import numpy as np

class CRF(nn.Module):
    '''
    Module implementing a conditional random field (CRF) output layer with the capacity for initializing transitions sensitive to the provided labeling scheme
    '''
    def __init__(self, classes, scheme, batch_first):
        '''
        Initializes the CRF module
            Arguments:
                classes: A list of classes (labels)
                scheme: The labeling scheme e.g. IOB1, IOB2, or IOBES
                batch_first: If True, the batch index is the first dimension as opposed to the last
            Returns:
                CRF module object
        '''
        super().__init__()
        # classes
        self.classes = classes
        # class prefixes
        self.prefixes = set([class_.split('-')[0] for class_ in self.classes])
        # labeling scheme
        self.scheme = scheme
        # initialize CRF
        self.crf = torchcrf.CRF(num_tags=len(self.classes), batch_first=batch_first)
    

    def initialize(self, seed):
        '''
        Initializes the CRF output layer
            Arguments:
                seed: Random seed for parameter initialization
            Returns:
                None
        '''
        # set seeds
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        # initialize weights
        self.crf.reset_parameters()
        if len(self.classes) > 1:
            # construct definitions of invalid transitions
            self.define_invalid_crf_transitions()
            # initialize transitions
            self.init_crf_transitions()
    

    def define_invalid_crf_transitions(self):
        '''
        Establishes valid tagging transitions for the given labeling scheme
            Arguments:
                None
            Returns:
                None
        '''
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
    

    def init_crf_transitions(self, penalty=-10000):
        '''
        Initializes CRF transitions according to invalid transitions as dictating by the labeling scheme
            Arguments:
                penalty: The static penalty to the loss for an invalid transition
            Returns:
                None
        '''
        num_tags = len(self.classes)
        # penalize bad beginnings and endings
        for i in range(num_tags):
            class_ = self.classes[i]
            if class_.split('-')[0] in self.invalid_begin:
                torch.nn.init.constant_(self.crf.start_transitions[i], penalty)
            if class_.split('-')[0] in self.invalid_end:
                torch.nn.init.constant_(self.crf.end_transitions[i], penalty)
        # build label type dictionary
        label_is = {}
        for label_position in self.prefixes:
            label_is[label_position] = [i for i, class_ in enumerate(self.classes) if class_.split('-')[0] == label_position]
        # penalties for invalid consecutive labels by position
        for from_label, to_label_list in self.invalid_transitions_position.items():
            to_labels = list(to_label_list)
            for from_label_i in label_is[from_label]:
                for to_label in to_labels:
                    for to_label_i in label_is[to_label]:
                        torch.nn.init.constant_(self.crf.transitions[from_label_i, to_label_i], penalty)
        # penalties for invalid consecutive labels by label
        for from_label, to_label_list in self.invalid_transitions_tags.items():
            to_labels = list(to_label_list)
            for from_label_i in label_is[from_label]:
                for to_label in to_labels:
                    for to_label_i in label_is[to_label]:
                        if self.classes[from_label_i].split('-')[1] != self.classes[to_label_i].split('-')[1]:
                            torch.nn.init.constant_(self.crf.transitions[from_label_i, to_label_i], penalty)
    

    def decode(self, emissions, mask):
        '''
        Decodes emmissions (logits) given a mask using a Viterbi decoder
            Arguments:
                emissions: Sequence logits
                mask: Mask for valid classification targets
            Returns:
                Most probable output sequence
        '''
        # verterbi decode logits (emissions) using valid attention mask
        crf_out = self.crf.decode(emissions, mask=mask)
        return crf_out


    def forward(self, emissions, labels, mask, reduction='token_mean'):
        '''
        Calculates the CRF loss given emissions (logits), the ground truth labels, masks, and the chosen reduction scheme
            Arguments:
                emissions: Sequence logits
                labels: Sequence ground truth labels
                mask: Mask for valid classification targets
                reduction: Loss averaging scheme
            Returns:
                CRF loss
        '''
        # calculate loss with forward pass of crf given logits (emissions) and valid attention mask
        # loss is mean over tokens
        crf_loss = self.crf(emissions, tags=labels, mask=mask, reduction=reduction)
        return crf_loss
