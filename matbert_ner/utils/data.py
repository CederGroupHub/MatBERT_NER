import json
from chemdataextractor.doc import Paragraph
from transformers import BertTokenizer
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

class NERData():

    def __init__(self, model_file="allenai/scibert_scivocab_uncased", scheme='IOB2'):
        # load tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(model_file)
        # initialize classes
        self.classes = None
        # invalid annotations (incomplete in solid_state file)
        self.invalid_annotations = ['PVL', 'PUT']
        # bert token limit
        self.token_limit = 512
        # minimum number of special tokens ([CLS] at beginning and [SEP] at end)
        self.special_token_count = 2
        # dictionaries of special tokens for fill values in both text and label fields
        self.pad_dict = {'text': '[PAD]', 'label': 'O'}
        self.unk_dict = {'text': '[UNK]', 'label': 'O'}
        self.sep_dict = {'text': '[SEP]', 'label': 'O'}
        self.cls_dict = {'text': '[CLS]', 'label': 'O'}
        # labeling scheme
        self.scheme = scheme
        # initialize dataset and dataloaders
        self.dataset = None
        self.dataloaders = None
    

    def get_classes(self, labels):
        # the raw classes are the provided labels
        classes_raw = labels
        # prefixes for labeling schemes
        if self.scheme in ['IOB', 'IOB2']:
            prefixes = ['I', 'B']
        elif self.scheme == 'IOBES':
            prefixes = ['B', 'I', 'E', 'S']
        # fill out labels with prefixes
        classes = ['{}-{}'.format(p, c) for p in prefixes for c in classes_raw if c not in self.invalid_annotations]
        # sort labels alphabetically
        classes = sorted(classes)
        # prepend 'O' label and set attribute
        self.classes = ['O']+classes

    
    def load_from_file(self, data_file):
        # list of entry identifiers
        identifiers = []
        # list of raw data (json entries)
        data_raw = []
        # set of raw labels
        labels = set([])
        # open data file
        with open(data_file, 'r') as f:
            # for line in file
            for l in f:
                # load json entry
                d = json.loads(l)
                # retrieve identifier (depends on the dataset, falls back to doi or doi+par)
                if 'solid_state' in data_file:
                    identifier = d['doi']
                elif 'aunp' in data_file:
                    identifier = d['meta']['doi']+'/'+str(d['meta']['par'])
                elif 'doping' in data_file:
                    identifier = d['text']
                else:
                    try:
                        identifier = d['doi']
                    except:
                        identifier = d['meta']['doi']+'/'+str(d['meta']['par'])
                # only entries with unique identifiers are retrieved
                if identifier in identifiers:
                    pass
                else:
                    identifiers.append(identifier)
                    data_raw.append(d)
                    # add labels in entry to raw label set
                    for l in d['labels']:
                        labels.add(l)
        # fill out classes
        self.get_classes(labels)
        return data_raw
    

    def shuffle_data(self, data, seed=256):
        # sets seed and shuffles if seed provided. 0 seed or None seed is actually unseeded
        if seed:
            random.Random(seed).shuffle(data)
        else:
            random.shuffle(data)
        return data
    

    def split_entries(self, data_raw, split_dict={'main': 1}, shuffle=False, seed=256):
        # shuffle if specified
        if shuffle:
            data_raw = self.shuffle_data(data_raw, seed)
        # retrieve keys from split dictionary
        split_keys = list(split_dict.keys())
        # fill list of split values
        split_vals = [split_dict[key] for key in split_keys]
        # calculate ending indices for splits based on size of dataset
        index_split_vals = (np.cumsum(split_vals)*len(data_raw)).astype(np.uint16)
        # split data according to split indices
        data_split = {split_keys[i]: data_raw[:index_split_vals[i]] if i == 0 else data_raw[index_split_vals[i-1]:index_split_vals[i]] for i in range(len(split_keys))}
        return data_split
    

    def format_entries(self, data_split):
        # initialize empty dictionary
        data_formatted = {split: [] for split in data_split.keys()}
        # for split in dataset
        for split in data_split.keys():
            # for entry in split
            for d in data_split[split]:
                # represent entry as list of dictionaries (sentences) with text and annotation keys for lists of the corresponding token properties
                data_formatted[split].append([{key: [token[key] for token in sentence] for key in ['text', 'annotation']} for sentence in d['tokens']])
        return data_formatted


    def label_entries(self, data_formatted):
        data_labeled = {split: [] for split in data_formatted.keys()}
        for split in data_formatted.keys():
            for dat in data_formatted[split]:
                d = []
                for sent in dat:
                    s = {key: [] for key in ['text', 'label']}
                    for i in range(len(sent['text'])):
                        if sent['text'][i] in ['̄','̊']:
                            continue
                        s['text'].append(sent['text'][i])
                        if self.scheme == 'IOB1':
                            if sent['annotation'][i] in [None, *self.invalid_annotations]:
                                s['label'].append('O')
                            elif i == 0 and len(sent['annotation']) > 1:
                                if sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('B-'+sent['annotation'][i])
                                else:
                                    s['label'].append('I-'+sent['annotation'][i])
                            elif i == 0 and len(sent['annotation']) == 1:
                                s['label'].append('I-'+sent['annotation'][i])
                            elif i > 0:
                                if sent['annotation'][i-1] == sent['annotation'][i]:
                                    s['label'].append('I-'+sent['annotation'][i])
                                else:
                                    if sent['annotation'][i+1] == sent['annotation'][i]:
                                        s['label'].append('B-'+sent['annotation'][i])
                                    else:
                                        s['label'].append('I-'+sent['annotation'][i])
                        elif self.scheme == 'IOB2':
                            if sent['annotation'][i] in [None, *self.invalid_annotations]:
                                s['label'].append('O')
                            elif i == 0:
                                s['label'].append('B-'+sent['annotation'][i])
                            elif i > 0:
                                if sent['annotation'][i-1] == sent['annotation'][i]:
                                    s['label'].append('I-'+sent['annotation'][i])
                                else:
                                    s['label'].append('B-'+sent['annotation'][i])
                        elif self.scheme == 'IOBES':
                            if sent['annotation'][i] in [None, *self.invalid_annotations]:
                                s['label'].append('O')
                            elif i == 0 and len(sent['annotation']) > 1:
                                if sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('B-'+sent['annotation'][i])
                                else:
                                    s['label'].append('S-'+sent['annotation'][i])
                            elif i == 0 and len(sent['annotation']) == 1:
                                s['label'].append('S-'+sent['annotation'][i])
                            elif i > 0 and i < len(sent['annotation'])-1:
                                if sent['annotation'][i-1] != sent['annotation'][i] and sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('B-'+sent['annotation'][i])
                                elif sent['annotation'][i-1] == sent['annotation'][i] and sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('I-'+sent['annotation'][i])
                                elif sent['annotation'][i-1] == sent['annotation'][i] and sent['annotation'][i+1] != sent['annotation'][i]:
                                    s['label'].append('E-'+sent['annotation'][i])
                                if sent['annotation'][i-1] != sent['annotation'][i] and sent['annotation'][i+1] != sent['annotation'][i]:
                                    s['label'].append('S-'+sent['annotation'][i])
                            elif i == len(sent['annotation'])-1:
                                if sent['annotation'][i-1] == sent['annotation'][i]:
                                    s['label'].append('E-'+sent['annotation'][i])
                                if sent['annotation'][i-1] != sent['annotation'][i]:
                                    s['label'].append('S-'+sent['annotation'][i])
                    d.append(s)
                data_labeled[split].append(d)
        return data_labeled

    
    def split_into_sentences(self, data_labeled):
        data_tagged_sentences = {split: [] for split in data_labeled.keys()}
        for split in data_labeled.keys():
            for d in data_labeled[split]:
                for s in d:
                    data_tagged_sentences[split].append(s)
        return data_tagged_sentences
    

    def combine_into_paragraphs(self, data_labeled):
        data_tagged_paragraphs = {split: [] for split in data_labeled.keys()}
        for split in data_labeled.keys():
            for d in data_labeled[split]:
                p = {key: [] for key in ['text', 'label']}
                for i in range(len(d)):
                    for key in p.keys():
                        p[key].extend(d[i][key])
                        if i < len(d)-1:
                            p[key].append(self.sep_dict[key])
                data_tagged_paragraphs[split].append(p)
        return data_tagged_paragraphs
    

    def combine_or_split(self, data_labeled, sentence_level):
        if sentence_level:
            return self.split_into_sentences(data_labeled)
        else:
            return self.combine_into_paragraphs(data_labeled)
    

    def create_examples(self, data_labeled):
        data_example = {split: [] for split in data_labeled.keys()}
        for split in data_labeled.keys():
            for n, dat in enumerate(data_labeled[split]):
                example = InputExample(n, dat['text'], dat['label'])
                data_example[split].append(example)
        return data_example
    

    def create_features(self, data_example):
        class_dict = {_class: i for i, _class in enumerate(self.classes)}
        data_feature = {split: [] for split in data_example.keys()}
        for split in data_example.keys():
            for example in data_example[split]:
                d = {key: [] for key in ['tokens', 'labels', 'token_ids', 'label_ids', 'attention_mask', 'valid_mask']}
                for i in range(len(example.text)):
                    word_tokens = self.tokenizer.tokenize(example.text[i])
                    for j, word_token in enumerate(word_tokens):
                        d['tokens'].append(word_token)
                        d['token_ids'].append(self.tokenizer.convert_tokens_to_ids(word_token))
                        d['attention_mask'].append(1)
                        if j == 0:
                            d['valid_mask'].append(1)
                            d['labels'].append(example.label[i])
                            d['label_ids'].append(class_dict[example.label[i]])
                        else:
                            d['valid_mask'].append(0)
                            d['labels'].append('O')
                            d['label_ids'].append(class_dict['O'])
                if len(d['tokens']) > self.token_limit-self.special_token_count:
                    d['tokens'] = d['tokens'][:self.token_limit-self.special_token_count]
                    d['labels'] = d['labels'][:self.token_limit-self.special_token_count]
                    d['token_ids'] = d['token_ids'][:self.token_limit-self.special_token_count]
                    d['label_ids'] = d['label_ids'][:self.token_limit-self.special_token_count]
                    d['attention_mask'] = d['attention_mask'][:self.token_limit-self.special_token_count]
                    d['valid_mask'] = d['valid_mask'][:self.token_limit-self.special_token_count]
                d['tokens'].insert(0, self.cls_dict['text'])
                d['labels'].insert(0, self.cls_dict['label'])
                d['token_ids'].insert(0, self.tokenizer.convert_tokens_to_ids(self.cls_dict['text']))
                d['label_ids'].insert(0, class_dict[self.cls_dict['label']])
                d['attention_mask'].insert(0, 1)
                d['valid_mask'].insert(0, 1)
                if d['tokens'][-1] != self.sep_dict['text']:
                    d['tokens'].append(self.sep_dict['text'])
                    d['labels'].append(self.sep_dict['label'])
                    d['token_ids'].append(self.tokenizer.convert_tokens_to_ids(self.sep_dict['text']))
                    d['label_ids'].append(class_dict[self.sep_dict['label']])
                    d['attention_mask'].append(1)
                    d['valid_mask'].append(1)
                data_feature[split].append(d)
        max_length = 0
        for split in data_example.keys():
            for d in data_feature[split]:
                length = len(d['tokens'])
                if length > max_length:
                    max_length = length
        for split in data_example.keys():
            for d in data_feature[split]:
                length = len(d['tokens'])
                d['tokens'].extend((max_length-length)*[self.pad_dict['text']])
                d['labels'].extend((max_length-length)*[self.pad_dict['label']])
                d['token_ids'].extend((max_length-length)*[self.tokenizer.convert_tokens_to_ids(self.pad_dict['text'])])
                d['label_ids'].extend((max_length-length)*[class_dict[self.pad_dict['label']]])
                d['attention_mask'].extend((max_length-length)*[0])
                d['valid_mask'].extend((max_length-length)*[0])
        data_input_feature = {split: [] for split in data_feature.keys()}
        for split in data_feature.keys():
            for d in data_feature[split]:
                data_input_feature[split].append(InputFeatures(token_ids=d['token_ids'],
                                                               label_ids=d['label_ids'],
                                                               attention_mask=d['attention_mask'],
                                                               valid_mask=d['valid_mask']))
        return data_input_feature
    

    def create_datasets(self, data_input_feature):
        self.dataset = {}
        for split in data_input_feature.keys():
            token_ids = torch.tensor([f.token_ids for f in data_input_feature[split]], dtype=torch.long)
            label_ids = torch.tensor([f.label_ids for f in data_input_feature[split]], dtype=torch.long)
            attention_mask = torch.tensor([f.attention_mask for f in data_input_feature[split]], dtype=torch.long)
            valid_mask = torch.tensor([f.valid_mask for f in data_input_feature[split]], dtype=torch.long)
            self.dataset[split] = TensorDataset(token_ids, label_ids, attention_mask, valid_mask)
    

    def preprocess(self, data, split_dict={'main': 1}, is_file=True, sentence_level=False, shuffle=False, seed=256):
        if is_file:
            data = self.load_from_file(data)
        if shuffle:
            data = self.shuffle_data(data, seed)
        self.create_datasets(self.create_features(self.create_examples(self.combine_or_split(self.label_entries(self.format_entries(self.split_entries(data, split_dict, shuffle, seed))), sentence_level))))   
    

    def create_dataloaders(self, batch_size=32, shuffle=False, seed=256):
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        self.dataloaders = {}
        for split in self.dataset.keys():
            self.dataloaders[split] = DataLoader(self.dataset[split], batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, token_ids, label_ids, attention_mask, valid_mask):
        self.token_ids = token_ids
        self.label_ids = label_ids
        self.attention_mask = attention_mask
        self.valid_mask = valid_mask


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, id, text, label):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.id = id
        self.text = text
        self.label = label
