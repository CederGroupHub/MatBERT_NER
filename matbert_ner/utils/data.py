import json
from transformers import BertTokenizer
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from matbert_ner.utils.tokenizer import MaterialsTextTokenizer
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
        self.class_dict = None
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
        self.class_dict = {class_: i for i, class_ in enumerate(self.classes)}
    

    def filter_data(self, data):
        # list of entry identifiers
        identifiers = []
        # list of raw data (json entries)
        data_filt = []
        for entry in tqdm(data, desc='| filtering entries |'):
            try:
                identifier = entry['meta']['doi']+'/'+str(entry['meta']['par'])+'/'+str(entry['meta']['split'])
            except:
                try:
                    identifier = entry['meta']['doi']+'/'+str(entry['meta']['par'])
                except:
                    try:
                        identifier = entry['doi']
                    except:
                        identifier = entry['text']
            # only entries with unique identifiers are retrieved
            if identifier in identifiers:
                pass
            else:
                identifiers.append(identifier)
                data_filt.append(entry)
        return identifiers, data_filt

    
    def load_from_memory_annotated(self, data):
        '''
        Loads raw annotated JSON entries from memory. Also calls the get_classes function on the collected labels in the JSON entries
            Arguments:
                data: List of unannotated entries
            Returns:
                List of dictionaries corresponding to the JSON entries
        '''
        # list of raw data (json entries)
        data_raw = []
        # set of raw labels
        labels = set([])
        # filter data by unique identifiers
        _, data_filt = self.filter_data(data)
        id = 0
        for entry in tqdm(data_filt, desc='| loading annotated entries |'):
            d = {'id': id, 'tokens': entry['tokens']}
            data_raw.append(d)
            # add labels in entry to raw label set
            for l in entry['labels']:
                labels.add(l)
            id += 1
        # fill out classes
        self.get_classes(labels)
        return data_raw
    

    def load_from_memory_unannotated(self, data):
        '''
        Loads raw JSON unannotated entries from memory. Also calls the get_classes function on the collected labels in the JSON entries
            Arguments:
                data: List of unannotated entries
            Returns:
                List of dictionaries corresponding to the JSON entries
        '''
        # list of raw data (json entries)
        data_raw = []
        # filter data by unique identifiers
        _, data_filt = self.filter_data(data)
        id = 0
        for entry in tqdm(data_filt, desc='| pre-tokenizing unannotated entries |'):
            d = {'id': id, 'tokens': []}
            try:
                sents = entry['tokens']
            except:
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
            id += 1
        # fill out classes
        self.get_classes([])
        return data_raw

    
    def load_from_memory(self, data, annotated=True):
        if annotated:
            data_raw = self.load_from_memory_annotated(data)
        else:
            data_raw = self.load_from_memory_unannotated(data)
        return data_raw

    
    def load_from_file(self, data_file, annotated=True):
        '''
        Loads raw JSON unannotated entries from file. Also calls the get_classes function on the collected labels in the JSON entries
            Arguments:
                data_file: Path to data file
            Returns:
                List of dictionaries corresponding to the JSON entries
        '''
        # open data file
        try:
            with open(data_file, 'r') as f:
                content = f.read()
                entries = json.loads(content)
        except:
            with open(data_file, 'r') as f:
                entries = []
                for l in tqdm(f, desc='| loading entries from file |'):
                    entries.append(json.loads(l))
        data_raw = self.load_from_memory(entries, annotated)
        return data_raw


    def load(self, data, is_file, annotated):
        if is_file:
            data_raw = self.load_from_file(data, annotated)
        else:
            data_raw = self.load_from_memory(data, annotated)
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
                data_formatted[split].append({'id': d['id'], 'tokens': [{key: [token[key] for token in sentence] for key in ['text', 'annotation']} for sentence in d['tokens']]})
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
                d = {'id': dat['id'], 'tokens': []}
                # for sentence in entry
                for sent in dat['tokens']:
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
                    d['tokens'].append(s)
                # append the entry to the labeled data split
                data_labeled[split].append(d)
        self.data = data_labeled
        return data_labeled

    
    def insert_cls(self, d):
            # dictionary of classes (given class name, return index)
            d['tokens'].insert(0, self.cls_dict['text'])
            d['labels'].insert(0, self.cls_dict['label'])
            d['token_ids'].insert(0, self.tokenizer.convert_tokens_to_ids(self.cls_dict['text']))
            d['label_ids'].insert(0, self.class_dict[self.cls_dict['label']])
            d['attention_mask'].insert(0, 1)
            d['valid_mask'].insert(0, 1)


    def create_features(self, data_labeled):
        '''
        Converts the dictionary of InputExamples into InputFeatures
            Arguments:
                data_labeled: A dictionary of InputExamples e.g. {'split': [InputExample,...],...}
            Returns:
                A dictionary of InputFeatures e.g. {'split': [{'tokens': [...], 'labels': [...], 'token_ids': [...], 'label_ids': [...], 'attention_mask': [...], 'valid_mask': [...]},...],...}
        '''
        
        # initialize empty dictionary
        data_feature = {split: [] for split in data_labeled.keys()}
        # for split in dataset
        for split in data_labeled.keys():
            data_label_range = tqdm(data_labeled[split], desc='| writing {} features |'.format(split))
            # for example in dataset split
            for dat in data_label_range:
                # initialize empty dictionary for features
                d = {key: dat['id'] if key == 'id' else [] for key in ['id', 'tokens', 'labels', 'token_ids', 'label_ids', 'attention_mask', 'valid_mask']}
                # for sentence in example
                for i in range(len(dat['tokens'])):
                    s = {key: [] for key in ['tokens', 'labels', 'token_ids', 'label_ids', 'attention_mask', 'valid_mask']}
                    # for token in sentence
                    n_tokens = len(dat['tokens'][i]['text'])
                    for j in range(n_tokens):
                        # split token using bert tokenizer
                        word_tokens = self.tokenizer.tokenize(dat['tokens'][i]['text'][j])
                        # for subtoken from bert tokenization
                        for k, word_token in enumerate(word_tokens):
                            # append to tokens
                            s['tokens'].append(word_token)
                            # append to token ids using tokenizer
                            s['token_ids'].append(self.tokenizer.convert_tokens_to_ids(word_token))
                            # append to attention mask
                            s['attention_mask'].append(1)
                            # if the subtoken is the first for the original token
                            if k == 0:
                                # the subtoken is valid for classification
                                s['valid_mask'].append(1)
                                # append the label
                                s['labels'].append(dat['tokens'][i]['label'][j])
                                # append the label id using the class dictionary
                                s['label_ids'].append(self.class_dict[dat['tokens'][i]['label'][j]])
                            # if the subtoken is not the first for the original token
                            else:
                                # the subtoken is not valid for classification
                                s['valid_mask'].append(0)
                                # append outside label as a placeholder (will not be seen by classifier)
                                s['labels'].append('O')
                                # append outside label id as placeholder (will not be seen by classifier)
                                s['label_ids'].append(self.class_dict['O'])
                        # append [SEP] token to end of sentence
                        if j == len(dat['tokens'][i]['text'])-1:
                            s['tokens'].append(self.sep_dict['text'])
                            s['labels'].append(self.sep_dict['label'])
                            s['token_ids'].append(self.tokenizer.convert_tokens_to_ids(self.sep_dict['text']))
                            s['label_ids'].append(self.class_dict[self.sep_dict['label']])
                            s['attention_mask'].append(1)
                            s['valid_mask'].append(1)
                    d['tokens'].append(s['tokens'])
                    d['labels'].append(s['labels'])
                    d['token_ids'].append(s['token_ids'])
                    d['label_ids'].append(s['label_ids'])
                    d['attention_mask'].append(s['attention_mask'])
                    d['valid_mask'].append(s['valid_mask'])
                data_feature[split].append(d)
        return data_feature


    def split_entries_merge_sentences(self, data_feature, sentence_level):
        # initialize empty dictionary


        def partition(a, k, no_improvement_threshold=5, max_iterations=100):
            # one split 
            if k <= 1 or k >= len(a): return [(0, len(a))]
            # partition between
            pb = [int((i+1)*len(a)/k) for i in range(k-1)]
            # average height
            ah = np.sum(a)/k
            # best score
            bs = None
            # best boundaries
            bb = None
            # iteration count
            n = 0
            while True:
                # starting indices
                s = [0]+pb
                # ending indices
                e = pb+[len(a)]
                # boundaries
                b = [(s[i], e[i]) for i in range(k)]
                # partitions
                p = [a[b[i][0]:b[i][1]] for i in range(k)]
                # heights
                h = [sum(sp) for sp in p]
                # absolute height differences
                ahd = np.abs(ah-h)
                # worst partition index
                wpi = np.argmax(ahd)
                whd = ah-h[wpi]
                # update best so far
                if bs is None or abs(whd) < bs:
                    bs = abs(whd)
                    bb = b
                    # no improvement count
                    nin = 0
                else:
                    # increment no improvement count
                    nin += 1
                # termination condition
                if whd == 0 or nin > no_improvement_threshold or n > max_iterations:
                    # return best partition boundaries
                    return bb
                # increment iterations
                n += 1
                # move
                m = -1 if whd < 0 else 1
                # bound to move
                mb = 0 if wpi == 0\
                                else k-2 if wpi == k-1\
                                else wpi-1 if (whd < 0) ^ (h[wpi-1] > h[wpi+1])\
                                else wpi
                # direction
                d = -1 if mb < wpi else 1
                pb[mb] += m*d

        dat_split_feature = {split: [] for split in data_feature.keys()}
        for split in data_feature.keys():
            data_label_range = tqdm(data_feature[split], desc=('| splitting long {} paragraphs' if sentence_level else '| splitting {} paragraphs |').format(split))
            for dat in data_label_range:
                if sentence_level:
                    for i in range(len(dat['tokens'])):
                        d = {'id': dat['id'], 'pt': i}
                        d.update({key: dat[key][i] for key in ['tokens', 'labels', 'token_ids', 'label_ids', 'attention_mask', 'valid_mask']})
                        dat_split_feature[split].append(d)
                else:
                    slen = [len(s) for s in dat['tokens']]
                    tnum = sum(slen)
                    n_splits = int(np.ceil((self.token_limit-np.sqrt(self.token_limit**2-4*tnum))/2))
                    if n_splits > 1:
                        bounds = partition(slen, n_splits)
                        for i in range(n_splits):
                            d = {'id': dat['id'], 'pt': i}
                            d.update({key: [v for s in dat[key][bounds[i][0]:bounds[i][1]] for v in s] for key in ['tokens', 'labels', 'token_ids', 'label_ids', 'attention_mask', 'valid_mask']})
                            self.insert_cls(d)
                            dat_split_feature[split].append(d)
                    else:
                        d = {'id': dat['id'], 'pt': 0}
                        d.update({key: [v for s in dat[key] for v in s] for key in ['tokens', 'labels', 'token_ids', 'label_ids', 'attention_mask', 'valid_mask']})
                        self.insert_cls(d)
                        dat_split_feature[split].append(d)
        return dat_split_feature
    

    def pad_features(self, data_split_feature):
        dat_input_feature = {split: [] for split in data_split_feature.keys()}
        for split in data_split_feature.keys():
            max_length = 0
            for dat in data_split_feature[split]:
                length = len(dat['tokens'])
                if length > max_length:
                    max_length = length
            for dat in data_split_feature[split]:
                d = {key: dat[key] for key in dat.keys()}
                length = len(dat['tokens'])
                d['tokens'].extend((max_length-length)*[self.pad_dict['text']])
                d['labels'].extend((max_length-length)*[self.pad_dict['label']])
                d['token_ids'].extend((max_length-length)*[self.tokenizer.convert_tokens_to_ids(self.pad_dict['text'])])
                d['label_ids'].extend((max_length-length)*[self.class_dict[self.pad_dict['label']]])
                d['attention_mask'].extend((max_length-length)*[0])
                d['valid_mask'].extend((max_length-length)*[0])
                dat_input_feature[split].append(d)
        return dat_input_feature


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
            ids = torch.tensor([d['id'] for d in data_input_feature[split]], dtype=torch.long, device=torch.device('cpu'))
            pts = torch.tensor([d['pt'] for d in data_input_feature[split]], dtype=torch.uint8, device=torch.device('cpu'))
            token_ids = torch.tensor([d['token_ids'] for d in data_input_feature[split]], dtype=torch.long, device=torch.device('cpu'))
            label_ids = torch.tensor([d['label_ids'] for d in data_input_feature[split]], dtype=torch.uint8, device=torch.device('cpu'))
            attention_mask = torch.tensor([d['attention_mask'] for d in data_input_feature[split]], dtype=torch.bool, device=torch.device('cpu'))
            valid_mask = torch.tensor([d['valid_mask'] for d in data_input_feature[split]], dtype=torch.bool, device=torch.device('cpu'))
            # store as tensor dataset
            self.dataset[split] = TensorDataset(ids, pts, token_ids, label_ids, attention_mask, valid_mask)
    

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
        data = self.load(data, is_file, annotated)
        # shuffle the entries if shuffle is True
        if shuffle:
            data = self.shuffle_data(data, seed)
        # creat datasets
        self.create_datasets(self.pad_features(self.split_entries_merge_sentences(self.create_features(self.label_entries(self.format_entries(self.split_entries(data, split_dict, shuffle, seed)))), sentence_level)))  
    

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
