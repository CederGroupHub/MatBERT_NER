import json
from transformers import BertTokenizer
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from utils.tokenizer import MaterialsTextTokenizer
from pathlib import Path

class NERData():
    '''
    An object for handling NER data
    '''
    def __init__(self, model_file="allenai/scibert_scivocab_uncased", scheme='IOBES'):
        '''
        Initializes the NERData object
            Arguments:
                model_file: Path to pre-trained BERT model
                scheme: Labeling scheme
            Returns:
                NERData object
        '''
        # load tokenizer
        self.pre_tokenizer = MaterialsTextTokenizer(Path(__file__).resolve().parent.as_posix()+'/phraser.pkl')
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
        self.data = None
        self.dataset = None
        self.dataloaders = None
    

    def get_classes(self, labels):
        '''
        Retrieves classes given raw labels using the labeling scheme. Saves classes as attribute.
            Arguments:
                labels: List of raw labels
            Returns:
                None
        '''
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

    
    def load_from_file_annotated(self, data_file):
        '''
        Loads raw annotated JSON entries from file. Also calls the get_classes function on the collected labels in the JSON entries
            Arguments:
                data_file: Path to data file
            Returns:
                List of dictionaries corresponding to the JSON entries
        '''
        # list of entry identifiers
        identifiers = []
        # list of raw data (json entries)
        data_raw = []
        # set of raw labels
        labels = set([])
        # open data file
        with open(data_file, 'r') as f:
            # for line in file
            for l in tqdm(f, desc='| loading annotated entries |'):
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

    
    def load_from_file_unannotated(self, data_file):
        '''
        Loads raw JSON unannotated entries from file. Also calls the get_classes function on the collected labels in the JSON entries
            Arguments:
                data_file: Path to data file
            Returns:
                List of dictionaries corresponding to the JSON entries
        '''
        # list of entry identifiers
        identifiers = []
        # list of raw data (json entries)
        data_raw = []
        # open data file
        try:
            with open(data_file, 'r') as f:
                content = f.read()
                entries = json.loads(content)
        except:
            with open(data_file, 'r') as f:
                entries = []
                for l in tqdm(f, desc='| loading unannotated entries |'):
                    entries.append(json.loads(l))
        for entry in tqdm(entries, desc='| pre-tokenizing unannotated entries |'):
            # retrieve identifier (depends on the dataset, falls back to doi or doi+par)
            try:
                identifier = entry['doi']
            except:
                identifier = entry['meta']['doi']+'/'+str(entry['meta']['par'])
            # only entries with unique identifiers are retrieved
            if identifier in identifiers:
                pass
            else:
                identifiers.append(identifier)
                d = {'doi': identifier, 'tokens': []}
                try:
                    sents = [self.pre_tokenizer.process(sent, convert_number=False, normalize_materials=False) for sent in entry['sents']]
                except:
                    sents = [self.pre_tokenizer.process(sent, convert_number=False, normalize_materials=False) for sent in self.pre_tokenizer.tokenize(entry['text'], keep_sentences=True)]
                for tokens in sents:
                    s = []
                    for tok in tokens:
                        s.append({'text': tok, 'annotation': None})
                    d['tokens'].append(s)
                data_raw.append(d)
        # fill out classes
        self.get_classes([])
        return data_raw


    def load_from_file(self, data_file, annotated=True):
        if annotated:
            return self.load_from_file_annotated(data_file)
        else:
            return self.load_from_file_unannotated(data_file)
    

    def load_unannotated(self, data):
        '''
        Loads raw JSON unannotated entries from file. Also calls the get_classes function on the collected labels in the JSON entries
            Arguments:
                data: List of unannotated entries
            Returns:
                List of dictionaries corresponding to the JSON entries
        '''
        # list of entry identifiers
        identifiers = []
        # list of raw data (json entries)
        data_raw = []
        for entry in data:
            try:
                identifier = entry['doi']
            except:
                identifier = entry['meta']['doi']+'/'+str(entry['meta']['par'])
            # only entries with unique identifiers are retrieved
            if identifier in identifiers:
                pass
            else:
                identifiers.append(identifier)
                d = {'doi': identifier, 'tokens': []}
                try:
                    sents = [self.pre_tokenizer.process(sent, convert_number=False, normalize_materials=False) for sent in entry['sents']]
                except:
                    sents = [self.pre_tokenizer.process(sent, convert_number=False, normalize_materials=False) for sent in self.pre_tokenizer.tokenize(entry['text'], keep_sentences=True)]
                for tokens in sents:
                    s = []
                    for tok in tokens:
                        s.append({'text': tok, 'annotation': None})
                    d['tokens'].append(s)
                data_raw.append(d)
        # fill out classes
        self.get_classes([])
        return data_raw


    def shuffle_data(self, data, seed=256):
        '''
        Shuffles a given dataset according to the provided seed. Will not be seeded if the seed returns a False value
            Arguments:
                data: Data to be shuffled
                seed: Random seed
            Returns:
                Shuffled data
        '''
        # sets seed and shuffles if seed provided. 0 seed or None seed is actually unseeded
        if seed:
            random.Random(seed).shuffle(data)
        else:
            random.shuffle(data)
        return data
    

    def split_entries(self, data_raw, split_dict={'main': 1}, shuffle=False, seed=256):
        '''
        Splits entries in a dataset according to a provided dictionary of splits and proportions
            Arguments:
                data_raw: JSON data loaded into a python dictionary
                split_dict: Dictionary of splits and proprotions e.g. {'split_1': 0.1, 'split_2': 0.1, 'split_3': 0.8}
                shuffle: Boolean for whether the raw data is shuffled before it is split
                seed: Random seed for shuffling. Will not be seeded if the seed returns a False value
            Returns:
                Dictionary of entries e.g. {'split_1': [...], 'split_2': [...], ...}
        '''
        # shuffle if specified
        if shuffle:
            data_raw = self.shuffle_data(data_raw, seed)
        # retrieve keys from split dictionary
        split_keys = list(split_dict.keys())
        # fill list of split values
        split_vals = [split_dict[key] for key in split_keys]
        # calculate ending indices for splits based on size of dataset
        index_split_vals = (np.cumsum(split_vals)*len(data_raw)).astype(np.uint32)
        # split data according to split indices
        data_split = {split_keys[i]: data_raw[:index_split_vals[i]] if i == 0 else data_raw[index_split_vals[i-1]:index_split_vals[i]] for i in range(len(split_keys))}
        return data_split
    

    def format_entries(self, data_split):
        '''
        Formats entries such that each consists of a list of sentences, each with a dictionary of text and annotations
            Arguments:
                data_split: A dictionary of data with the splits as the keys
            Returns:
                Formatted entries in the form {'split': [[[{'text': [...], 'annotation': [...]}],...],...],...}
        '''
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
        '''
        Labels entries according to the desired labeling scheme
            Arguments:
                data_formatted: A dictionary of formatted data with the splits as the keys e.g. {'split': [[[{'text': [...], 'annotation': [...]}],...],...],...}
            Returns:
                Labeled data of same format as input, but the 'annotations' field for each sentence is replaced with a 'label' field where a label is in the form <Prefix>-<Annotation>
        '''
        # initialize empty dictionary
        data_labeled = {split: [] for split in data_formatted.keys()}
        # for split in dataset
        for split in data_formatted.keys():
            # for entry in split (paragraph)
            for dat in data_formatted[split]:
                # initialize empty list
                d = []
                # for sentence in entry
                for sent in dat:
                    # initialize text/label dictionary for sentence
                    s = {key: [] for key in ['text', 'label']}
                    # for token in sentence
                    for i in range(len(sent['text'])):
                        # skip tokens that don't work with bert
                        if sent['text'][i] in ['̄','̊']:
                            continue
                        # otherwise append token to sentence
                        s['text'].append(sent['text'][i])
                        # inside-outside-beginning scheme (1)
                        if self.scheme == 'IOB1':
                            # None or invalid annotations are mapped to outside
                            if sent['annotation'][i] in [None, *self.invalid_annotations]:
                                s['label'].append('O')
                            # if this is the first token in a sentence of more than one token
                            elif i == 0 and len(sent['annotation']) > 1:
                                # if the next token is of the same type, it is the beginning of an entity
                                if sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('B-'+sent['annotation'][i])
                                # otherwise inside
                                else:
                                    s['label'].append('I-'+sent['annotation'][i])
                            # if the sentence is only one token long and not outside, it must be inside
                            elif i == 0 and len(sent['annotation']) == 1:
                                s['label'].append('I-'+sent['annotation'][i])
                            # if the token is not the first in the sentence
                            elif i > 0:
                                # if the prior token was of the same type, it is inside
                                if sent['annotation'][i-1] == sent['annotation'][i]:
                                    s['label'].append('I-'+sent['annotation'][i])
                                else:
                                    # if the prior token was of a different type and the next is of the same type, beginning
                                    if sent['annotation'][i+1] == sent['annotation'][i]:
                                        s['label'].append('B-'+sent['annotation'][i])
                                    # otherwise inside
                                    else:
                                        s['label'].append('I-'+sent['annotation'][i])
                        # inside-outside-beginning scheme (2)
                        elif self.scheme == 'IOB2':
                            # None or invalid annotations are mapped to outside
                            if sent['annotation'][i] in [None, *self.invalid_annotations]:
                                s['label'].append('O')
                            # if the first token is an entity, it must be the beginning
                            elif i == 0:
                                s['label'].append('B-'+sent['annotation'][i])
                            # if the token is not the first in the sentence
                            elif i > 0:
                                # if the prior token was of the same type, then it is inside
                                if sent['annotation'][i-1] == sent['annotation'][i]:
                                    s['label'].append('I-'+sent['annotation'][i])
                                # otherwise, it is beginning
                                else:
                                    s['label'].append('B-'+sent['annotation'][i])
                        # inside-outside-beginning-end-single scheme
                        elif self.scheme == 'IOBES':
                            # None or invalid annotations are mapped to outside
                            if sent['annotation'][i] in [None, *self.invalid_annotations]:
                                s['label'].append('O')
                            # if the the single token in a sentence is an entity, it must be single
                            elif i == 0 and len(sent['annotation']) == 1:
                                s['label'].append('S-'+sent['annotation'][i])
                            # if the first token in a multi-token sentence is an entity
                            elif i == 0 and len(sent['annotation']) > 1:
                                # if the next token is of the same type, it is beginning
                                if sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('B-'+sent['annotation'][i])
                                # if the next token is not of the same type, it is single
                                else:
                                    s['label'].append('S-'+sent['annotation'][i])
                            # if not the first or last token
                            elif i > 0 and i < len(sent['annotation'])-1:
                                # if the token before is of a different type and the next of the same, then beginning
                                if sent['annotation'][i-1] != sent['annotation'][i] and sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('B-'+sent['annotation'][i])
                                # if the token before is of the same type and the next of the same, then inside
                                elif sent['annotation'][i-1] == sent['annotation'][i] and sent['annotation'][i+1] == sent['annotation'][i]:
                                    s['label'].append('I-'+sent['annotation'][i])
                                # if the token before is of the same type and the next of a different, then end
                                elif sent['annotation'][i-1] == sent['annotation'][i] and sent['annotation'][i+1] != sent['annotation'][i]:
                                    s['label'].append('E-'+sent['annotation'][i])
                                # if the token before is of a different type and the next of a different, then single
                                elif sent['annotation'][i-1] != sent['annotation'][i] and sent['annotation'][i+1] != sent['annotation'][i]:
                                    s['label'].append('S-'+sent['annotation'][i])
                            # if the last token
                            elif i == len(sent['annotation'])-1:
                                # if the token before is of the same type, then end
                                if sent['annotation'][i-1] == sent['annotation'][i]:
                                    s['label'].append('E-'+sent['annotation'][i])
                                # if the token before is of a different type, then single
                                if sent['annotation'][i-1] != sent['annotation'][i]:
                                    s['label'].append('S-'+sent['annotation'][i])
                    # append the labeled sentence to the entry
                    d.append(s)
                # append the entry to the labeled data split
                data_labeled[split].append(d)
        self.data = data_labeled
        return data_labeled

    
    def split_into_sentences(self, data_labeled):
        '''
        Splits entries into sentences as separate entries
            Arguments:
                data_labeled: A dictionary of split labeled entries e.g. {'split': [[[{'text': [...], 'label': [...]}],...],...],...}
            Returns:
                A dictionary of entries by sentence rather than paragraph then sentence e.g. {'split': [[{'text': [...], 'label': [...]}],...],...}
        '''
        # initialize empty dictionary
        data_labeled_sentences = {split: [] for split in data_labeled.keys()}
        # for split in dataset
        for split in data_labeled.keys():
            # for entry in dataset split
            for d in data_labeled[split]:
                # for sentence in entry
                for s in d:
                    # append sentence to dictionary
                    data_labeled_sentences[split].append(s)
        return data_labeled_sentences
    

    def combine_into_paragraphs(self, data_labeled):
        '''
        Combines the sentences separated by [SEP] tokens within each entry into a single sequence
            Arguments:
                data_labeled: A dictionary of split labeled entries e.g. {'split': [[[{'text': [...], 'label': [...]}],...],...],...}
            Returns:
                A dictionary of entries by paragraph with sentences combined rather than paragraph then sentence e.g. {'split': [[{'text': [...], 'label': [...]}],...],...}
        '''
        # initialize empty dictionary
        data_labeled_paragraphs = {split: [] for split in data_labeled.keys()}
        # for split in dataset
        for split in data_labeled.keys():
            # for entry in dataset split
            for d in data_labeled[split]:
                # initialize empty paragraph dictionary
                p = {key: [] for key in ['text', 'label']}
                # for sentence in entry
                for i in range(len(d)):
                    # for key in paragraph dictionary
                    for key in p.keys():
                        # extend paragraph with keyed sentence
                        p[key].extend(d[i][key])
                        # if the sentence is not the last
                        if i < len(d)-1:
                            # append the appropriate value for the [SEP] token
                            p[key].append(self.sep_dict[key])
                # append paragraph to dictionary
                data_labeled_paragraphs[split].append(p)
        return data_labeled_paragraphs
    

    def combine_or_split(self, data_labeled, sentence_level):
        '''
        Combines the sentences within an entry into a single sequence or splits the sentences into separate entries
            Arguments:
                data_labelled: A dictionary of split labeled entries e.g. {'split': [[[{'text': [...], 'label': [...]}],...],...],...}
                sentence_level: Boolean that controls whether the sentences in entries are split into separate entries (True) or combines them into a single sequence entry (False)
            Returns:
                A dictionary of single-sequence entries e.g. {'split': [[{'text': [...], 'label': [...]}],...],...}
        '''
        # either combine sentences into single entry or split sentences into separate entries
        if sentence_level:
            return self.split_into_sentences(data_labeled)
        else:
            return self.combine_into_paragraphs(data_labeled)
    

    def create_features(self, data_labeled):
        '''
        Converts the dictionary of InputExamples into InputFeatures
            Arguments:
                data_labeled: A dictionary of InputExamples e.g. {'split': [InputExample,...],...}
            Returns:
                A dictionary of InputFeatures e.g. {'split': [{'tokens': [...], 'labels': [...], 'token_ids': [...], 'label_ids': [...], 'attention_mask': [...], 'valid_mask': [...]},...],...}
        '''
        # dictionary of classes (given class name, return index)
        class_dict = {class_: i for i, class_ in enumerate(self.classes)}
        # initialize empty dictionary
        data_feature = {split: [] for split in data_labeled.keys()}
        # for split in dataset
        for split in data_labeled.keys():
            data_label_range = tqdm(data_labeled[split], desc='| writing {} features |'.format(split))
            # for example in dataset split
            for dat in data_label_range:
                # initialize empty dictionary for features
                d = {key: [] for key in ['tokens', 'labels', 'token_ids', 'label_ids', 'attention_mask', 'valid_mask']}
                # for token in example
                for i in range(len(dat['text'])):
                    # split token using bert tokenizer
                    word_tokens = self.tokenizer.tokenize(dat['text'][i])
                    # for subtoken from bert tokenization
                    for j, word_token in enumerate(word_tokens):
                        # append to tokens
                        d['tokens'].append(word_token)
                        # append to token ids using tokenizer
                        d['token_ids'].append(self.tokenizer.convert_tokens_to_ids(word_token))
                        # append to attention mask
                        d['attention_mask'].append(1)
                        # if the subtoken is the first for the original token
                        if j == 0:
                            # the subtoken is valid for classification
                            d['valid_mask'].append(1)
                            # append the label
                            d['labels'].append(dat['label'][i])
                            # append the label id using the class dictionary
                            d['label_ids'].append(class_dict[dat['label'][i]])
                        # if the subtoken is not the first for the original token
                        else:
                            # the subtoken is not valid for classification
                            d['valid_mask'].append(0)
                            # append outside label as a placeholder (will not be seen by classifier)
                            d['labels'].append('O')
                            # append outside label id as placeholder (will not be seen by classifier)
                            d['label_ids'].append(class_dict['O'])
                # if the entry has exceeded the token limit (accounting for the special tokens)
                if len(d['tokens']) > self.token_limit-self.special_token_count:
                    # truncate the lists
                    d['tokens'] = d['tokens'][:self.token_limit-self.special_token_count]
                    d['labels'] = d['labels'][:self.token_limit-self.special_token_count]
                    d['token_ids'] = d['token_ids'][:self.token_limit-self.special_token_count]
                    d['label_ids'] = d['label_ids'][:self.token_limit-self.special_token_count]
                    d['attention_mask'] = d['attention_mask'][:self.token_limit-self.special_token_count]
                    d['valid_mask'] = d['valid_mask'][:self.token_limit-self.special_token_count]
                # insert the cls token at the beginning of each list
                d['tokens'].insert(0, self.cls_dict['text'])
                d['labels'].insert(0, self.cls_dict['label'])
                d['token_ids'].insert(0, self.tokenizer.convert_tokens_to_ids(self.cls_dict['text']))
                d['label_ids'].insert(0, class_dict[self.cls_dict['label']])
                d['attention_mask'].insert(0, 1)
                d['valid_mask'].insert(0, 1)
                # if the last token is not a [SEP] token
                if d['tokens'][-1] != self.sep_dict['text']:
                    # append the sep token at the end of each list
                    d['tokens'].append(self.sep_dict['text'])
                    d['labels'].append(self.sep_dict['label'])
                    d['token_ids'].append(self.tokenizer.convert_tokens_to_ids(self.sep_dict['text']))
                    d['label_ids'].append(class_dict[self.sep_dict['label']])
                    d['attention_mask'].append(1)
                    d['valid_mask'].append(1)
                # append entry to features dictionary
                data_feature[split].append(d)
        # initialize maximum entry length
        max_length = 0
        # for split in dataset
        for split in data_labeled.keys():
            # for entry in dataset split
            for d in data_feature[split]:
                # length of entry
                length = len(d['tokens'])
                # update maximum entry length if necessary
                if length > max_length:
                    max_length = length
        # for split in datset
        for split in data_labeled.keys():
            # for entry in datset split
            for d in data_feature[split]:
                # length of entry
                length = len(d['tokens'])
                # pad entries to maximum length
                d['tokens'].extend((max_length-length)*[self.pad_dict['text']])
                d['labels'].extend((max_length-length)*[self.pad_dict['label']])
                d['token_ids'].extend((max_length-length)*[self.tokenizer.convert_tokens_to_ids(self.pad_dict['text'])])
                d['label_ids'].extend((max_length-length)*[class_dict[self.pad_dict['label']]])
                d['attention_mask'].extend((max_length-length)*[0])
                d['valid_mask'].extend((max_length-length)*[0])
        return data_feature
    

    def create_datasets(self, data_input_feature):
        '''
        Creates datsets from a dictionary of InputFeatures, which are saved as an attribute
            Arguments:
                data_input_feature: A dictionary of InputFeatures e.g. {'split': [InputFeatures,...],...}
            Returns:
                None
        '''
        # initialize empty dictionary
        self.dataset = {}
        # for split in dataset
        for split in data_input_feature.keys():
            # collect features
            token_ids = torch.tensor([d['token_ids'] for d in data_input_feature[split]], dtype=torch.long, device=torch.device('cpu'))
            label_ids = torch.tensor([d['label_ids'] for d in data_input_feature[split]], dtype=torch.uint8, device=torch.device('cpu'))
            attention_mask = torch.tensor([d['attention_mask'] for d in data_input_feature[split]], dtype=torch.bool, device=torch.device('cpu'))
            valid_mask = torch.tensor([d['valid_mask'] for d in data_input_feature[split]], dtype=torch.bool, device=torch.device('cpu'))
            # store as tensor dataset
            self.dataset[split] = TensorDataset(token_ids, label_ids, attention_mask, valid_mask)
    

    def preprocess(self, data, split_dict={'main': 1}, is_file=True, annotated=True, sentence_level=False, shuffle=False, seed=256):
        '''
        Preprocesses raw data provided in either dictionary or JSON form to produce datasets which are saved as an attribute
            Arguments:
                data: Either a dictionary of raw entries or the path to a JSON file containing raw entries
                split_dict: Dictionary of splits and proprotions e.g. {'split_1': 0.1, 'split_2': 0.1, 'split_3': 0.8}
                is_file: Boolean that controls whether data is treated as file (True) or list (False)
                sentence_level: Boolean that controls whether the sentences in entries are split into separate entries (True) or combines them into a single sequence entry (False)
                shuffle: Boolean for whether the raw data is shuffled before it is split
                seed: Random seed for shuffling. Will not be seeded if the seed returns a False value
            Returns:
                None
        '''
        # call load from file if the data is a file
        if is_file:
            data = self.load_from_file(data, annotated)
        elif not annotated:
            data = self.load_unannotated(data)
        # shuffle the entries if shuffle is True
        if shuffle:
            data = self.shuffle_data(data, seed)
        # creat datasets
        self.create_datasets(self.create_features(self.combine_or_split(self.label_entries(self.format_entries(self.split_entries(data, split_dict, shuffle, seed))), sentence_level)))  
    

    def create_dataloaders(self, batch_size=32, shuffle=True, seed=256):
        '''
        Creates dataloaders from dictionary of datasets which are saved as an attribute
            Arguments:
                batch_size: Number of entries per batch in the dataloaders
                shuffle: Boolean that controls whether the data is shuffled within the dataloaders between training epochs
                seed: Random seed for shuffling. Will not be seeded if the seed returns a False value
            Return:
                None
        '''
        # set seeds if a seed is provided
        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        # initialize empty dictionary
        self.dataloaders = {}
        # for split in dataset
        for split in self.dataset.keys():
            # store dataloaders for tensor datasets
            self.dataloaders[split] = DataLoader(self.dataset[split], batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)
