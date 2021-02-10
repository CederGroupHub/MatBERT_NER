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
        # no_decay = ["bias", "LayerNorm.weight"]

        # bert_parameters = self.model.bert.named_parameters()
        # classifier_parameters = self.model.classifier.named_parameters()
        # bert_lr = self.lr
        # classifier_lr = self.lr
        # optimizer_grouped_parameters = [
        #     {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0,
        #      "lr": bert_lr},
        #     {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0,
        #      "lr": bert_lr},

        #     {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0,
        #      "lr": classifier_lr},
        #     {"params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
        #      "weight_decay": 0.0,
        #      "lr": classifier_lr}
        # ]
        optimizer = optim.AdamW(self.model.parameters(), self.lr, eps=1e-8)
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
        sequence_output, attention_mask = valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, self.device)
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
        self.new_crf = True
        self.bert = BertModel(config)
        self._device = device
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        if self.new_crf:
            self.crf = CRF_NEW(tag_names=tag_names, batch_first=True)
        else:
            self.crf = CRF(tag_names=tag_names, batch_first=True)
        self.init_weights()
        if self.new_crf:
            self.crf.define_invalid_crf_transitions()
            self.crf.init_crf_transitions()

    
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
        sequence_output, attention_mask = valid_sequence_output(input_ids, sequence_output, valid_mask, attention_mask, self.device)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        if decode:
            tags = self.crf.decode(logits, mask=attention_mask)
            outputs = (tags,)
        else:
            outputs = (logits,)

        if labels is not None:
            labels = torch.where(labels >= 0, labels, torch.zeros_like(labels))
            loss = self.crf(logits, labels, mask=attention_mask)
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
                # if input_ids[i][j].item() in [102, 103] and crf:
                #     valid_attention_mask[i, jj] = 0
                # else:
                valid_attention_mask[i][jj] = attention_mask[i][j]
    return valid_output, valid_attention_mask


class CRF_NEW(nn.Module):
    def __init__(self, tag_names, batch_first):
        super().__init__()
        # tag pad index and tag names
        self.pad_token = '[PAD]'
        self.tag_names = tag_names
        # initialize CRF
        self.crf = torchcrf.CRF(num_tags=len(self.tag_names), batch_first=batch_first)
        # construct definitions of invalid transitions
        self.define_invalid_crf_transitions()
        # initialize transitions
        self.init_crf_transitions()
    

    def define_invalid_crf_transitions(self):
        ''' function for establishing valid tagging transitions, assumes BIO or BILUO tagging '''
        self.prefixes = set([tag_name[0] for tag_name in self.tag_names if tag_name != self.pad_token])
        if self.prefixes == set(['B', 'I', 'O']):
            # (B)eginning (I)nside (O)utside
            # sentence must begin with [CLS] (O)
            self.invalid_begin = ('B', 'I')
            # self.invalid_begin = ('I',)
            # sentence must end with [SEP] (O)
            self.invalid_end = ('B', 'I')
            # self.invalid_end = ('B', "I")
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
            # sentence must begin with [CLS] (O)
            self.invalid_begin = ('B', 'I', 'L', 'U')
            # self.invalid_begin = ('I', 'L')
            # sentence must end with [SEP] (O)
            self.invalid_end = ('B', 'I', 'L', 'U')
            # self.invalid_end = ('B', "I")
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
            # sentence must begin with [CLS] (O)
            self.invalid_begin = ('B', 'I', 'E', 'S')
            # self.invalid_begin = ('I', 'E')
            # sentence must end with [SEP] (O)
            self.invalid_end = ('B', 'I', 'E', 'S')
            # self.invalid_end = ('B', "I")
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
            if tag_name[0] in self.invalid_begin:
                torch.nn.init.constant_(self.crf.start_transitions[i], imp_value)
            if tag_name[0] in self.invalid_end:
                torch.nn.init.constant_(self.crf.end_transitions[i], imp_value)
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
    

    def decode(self, emissions, mask):
        crf_out = self.crf.decode(emissions, mask=mask)
        return crf_out


    def forward(self, emissions, tags, mask):
        crf_loss = self.crf(emissions, tags=tags, mask=mask)
        return crf_loss


class CRF(nn.Module):
    def __init__(self, tag_names: list, batch_first: bool = False) -> None:
        num_tags = len(tag_names)
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.
        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum'):
        """Compute the conditional log likelihood of a sequence of tags given emission scores.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.
        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        return self._forward_alg(emissions, tags, mask, reduction)

    def _forward_alg(self,
                     emissions: torch.Tensor,
                     tags: torch.LongTensor,
                     mask: Optional[torch.ByteTensor] = None,
                     reduction: str = 'sum',
                     ) -> torch.Tensor:
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.
        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].bool().all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].bool().all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].bool().all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].bool().all()
        mask = mask.bool()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].bool().all()
        mask = mask.bool()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)
        best_tags_list = [item + [-1] * (seq_length - len(item)) for item in best_tags_list]
        best_tags_list = torch.from_numpy(np.array(best_tags_list))
        return torch.LongTensor(best_tags_list).cuda()