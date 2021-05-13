from transformers import BertTokenizer
from chemdataextractor.doc import Paragraph
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, SequentialSampler, Subset, RandomSampler
import json
import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold

class NERData():

    def __init__(self, modelname="allenai/scibert_scivocab_cased", tag_scheme='IOB2'):
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.classes = None
        self.token_limit = 512
        self.special_token_count = 2
        self.tag_scheme = tag_scheme
        self.max_sequence_length = None
        self.dataset = None
        self.dataloaders = None

    
    def load_from_file(self, datafile):
        identifiers = []
        data = []
        labels = set([])
        with open(datafile, 'r') as f:
            for l in f:
                d = json.loads(l)
                if 'solid_state' in datafile:
                    identifier = d['doi']
                elif 'aunp' in datafile:
                    identifier = d['meta']['doi']
                elif 'doping' in datafile:
                    identifier = d['text']
                else:
                    try:
                        identifier = d['doi']
                    except:
                        identifier = d['meta']['doi']
                if identifier in identifiers:
                    pass
                else:
                    identifiers.append(identifier)
                    data.append(d)
                    for l in d['labels']:
                        labels.add(l)
        self.__get_classes(labels)
        return data
    

    def shuffle_data(self, data, seed=256):
        random.Random(seed).shuffle(data)
        return data
    

    def split_entries(self, data, splits, shuffle=True, seed=256):
        if shuffle:
            data = self.shuffle_data(data, seed)
        splits = (np.cumsum(splits)*len(data)).astype(np.uint16)
        test_set = data[:splits[0]]
        valid_set = data[splits[0]:splits[1]]
        train_set = data[splits[1]:splits[2]]
        data_split = {'test': test_set, 'valid': valid_set, 'train': train_set}
        return data_split
    

    def format_entries(self, data_split, shuffle=True, seed=256, sentence_level=True):
        data_fmt = {split: [] for split in data_split.keys()}
        for split in data_split.keys():
            for d in data_split[split]:
                if sentence_level:
                    dat = [{key: [token[key] for token in sentence][:self.token_limit-self.special_token_count] for key in ['text', 'annotation']} for sentence in d['tokens']]
                else:
                    dat = [{key : [token[key] for sentence in d['tokens'] for token in sentence][:self.token_limit-self.special_token_count] for key in ['text', 'annotation']}]
                data_fmt[split].extend(dat)
            if shuffle:
                data_fmt[split] = self.shuffle_data(data_fmt[split], seed)
        return data_fmt


    def tag_entries(self, data_fmt):
        data_tag = {split: [] for split in data_fmt.keys()}
        for split in data_fmt.keys():
            for dat in data_fmt[split]:
                d = {key: [] for key in ['text', 'tag']}
                for i in range(len(dat['text'])):
                    if dat['text'][i] in ['̄','̊']:
                        continue
                    d['text'].append(dat['text'][i])
                    if self.tag_scheme == 'IOB1':
                        if dat['annotation'][i] in [None, 'PVL', 'PUT']:
                            d['tag'].append('O')
                        elif i == 0 and len(dat['annotation']) > 1:
                            if dat['annotation'][i+1] == dat['annotation'][i]:
                                d['tag'].append('B-'+dat['annotation'][i])
                            else:
                                d['tag'].append('I-'+dat['annotation'][i])
                        elif i == 0 and len(dat['annotation']) == 1:
                            d['tag'].append('I-'+dat['annotation'][i])
                        elif i > 0:
                            if dat['annotation'][i-1] == dat['annotation'][i]:
                                d['tag'].append('I-'+dat['annotation'][i])
                            else:
                                if dat['annotation'][i+1] == dat['annotation'][i]:
                                    d['tag'].append('B-'+dat['annotation'][i])
                                else:
                                    d['tag'].append('I-'+dat['annotation'][i])
                    elif self.tag_scheme == 'IOB2':
                        if dat['annotation'][i] in [None, 'PVL', 'PUT']:
                            d['tag'].append('O')
                        elif i == 0:
                            d['tag'].append('B-'+dat['annotation'][i])
                        elif i > 0:
                            if dat['annotation'][i-1] == dat['annotation'][i]:
                                d['tag'].append('I-'+dat['annotation'][i])
                            else:
                                d['tag'].append('B-'+dat['annotation'][i])
                    elif self.tag_scheme == 'IOBES':
                        if dat['annotation'][i] in [None, 'PVL', 'PUT']:
                            d['tag'].append('O')
                        elif i == 0 and len(dat['annotation']) > 1:
                            if dat['annotation'][i+1] == dat['annotation'][i]:
                                d['tag'].append('B-'+dat['annotation'][i])
                            else:
                                d['tag'].append('S-'+dat['annotation'][i])
                        elif i == 0 and len(dat['annotation']) == 1:
                            d['tag'].append('S-'+dat['annotation'][i])
                        elif i > 0 and i < len(dat['annotation'])-1:
                            if dat['annotation'][i-1] != dat['annotation'][i] and dat['annotation'][i+1] == dat['annotation'][i]:
                                d['tag'].append('B-'+dat['annotation'][i])
                            elif dat['annotation'][i-1] == dat['annotation'][i] and dat['annotation'][i+1] == dat['annotation'][i]:
                                d['tag'].append('I-'+dat['annotation'][i])
                            elif dat['annotation'][i-1] == dat['annotation'][i] and dat['annotation'][i+1] != dat['annotation'][i]:
                                d['tag'].append('E-'+dat['annotation'][i])
                            if dat['annotation'][i-1] != dat['annotation'][i] and dat['annotation'][i+1] != dat['annotation'][i]:
                                d['tag'].append('S-'+dat['annotation'][i])
                        elif i == len(dat['annotation'])-1:
                            if dat['annotation'][i-1] == dat['annotation'][i]:
                                d['tag'].append('E-'+dat['annotation'][i])
                            if dat['annotation'][i-1] != dat['annotation'][i]:
                                d['tag'].append('S-'+dat['annotation'][i])
                data_tag[split].append(d)
        return data_tag

    
    def create_examples(self, data_tag):
        data_example = {split: [] for split in data_tag.keys()}
        self.max_sequence_length = 0
        for split in data_tag.keys():
            for n, dat in enumerate(data_tag[split]):
                sequence_length = len(dat['text'])
                if sequence_length > self.max_sequence_length:
                    self.max_sequence_length = sequence_length
                example = InputExample(n, dat['text'], dat['tag'])
                data_example[split].append(example)
        self.max_sequence_length += self.special_token_count
        return data_example
    

    def create_datasets(self, data_example):
        self.dataset = {}
        for split in data_example.keys():
            features = self.__convert_examples_to_features(data_example[split], split, self.classes, self.max_sequence_length)
            input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
            segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
            self.dataset[split] = TensorDataset(input_ids, input_mask, valid_mask, segment_ids, label_ids)
        return self
    

    def preprocess(self, datafile, splits, is_file=True, sentence_level=True, shuffle=True, seed=256):
        if is_file:
            data = self.load_from_file(datafile)
        else:
            data = datafile
        if shuffle:
            data = self.shuffle_data(data, seed)
        self.create_datasets(self.create_examples(self.tag_entries(self.format_entries(self.split_entries(data, splits, shuffle, seed), shuffle, seed, sentence_level))))
        return self        
    

    def create_dataloaders(self, batch_size=32):
        self.dataloaders = {}
        for split in self.dataset.keys():
            self.dataloaders[split] = DataLoader(self.dataset[split], batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        return self
    

    def create_tokenset(self, text):
        idx = 0
        tokens = Paragraph(text).tokens
        sentences = []
        sentence = []
        for sent in tokens:
            for tok in sent:
                tok_dict = {"text" : tok.text,
                            "start" : tok.start,
                            "end" : tok.end,
                            "annotation" : None}
                sentence.append(tok_dict)
            sentences.append(sentence)
            sentence = []
        tokenset = dict(text=text, tokens=sentences)
        cleaned_tokenset = self._clean_tokenset(tokenset)
        return cleaned_tokenset
    

    def _clean_tokenset(self, tokenset):
        bad_chars = ['\xa0', '\u2009', '\u202f', '\u200c', '\u2fff', 'ͦ', '\u2061', '\ue5f8']
        problem_child = False
        start = 0
        good_sentences = []
        for sent in tokenset['tokens']:
            good_toks = []
            for tok in sent:
                if tok['text'] not in bad_chars:
                    if tok['start'] == start:
                        good_toks.append(tok)
                    else:
                        tok['start'] = start
                        tok['end'] = start + len(tok['text'])
                        good_toks.append(tok)

                    start = tok['end'] + 1

                else:
                    problem_child = True
            good_sentences.append(good_toks)
            if problem_child:
                problem_child = False
        tokenset['tokens'] = good_sentences
        return tokenset


    def __convert_examples_to_features(self, examples, split, label_list, max_seq_length,
                                       cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=1,
                                       sep_token="[SEP]", sep_token_extra=False,
                                       pad_on_left=False, pad_token=0, pad_token_segment_id=0, pad_token_label_id=-100,
                                       sequence_a_segment_id=0, mask_padding_with_zero=True):
        """ Loads a data file into a list of `InputBatch`s
            `cls_token_at_end` define the location of the CLS token:
                - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
                - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
            `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        label_map = {label: i for i, label in enumerate(label_list)}
        span_labels = []
        for label in label_list:
            label = label.split('-')[-1]
            if label not in span_labels:
                span_labels.append(label)
        span_map = {label: i for i, label in enumerate(span_labels)}
        features = []
        example_range = tqdm(examples, desc='| writing {} |'.format(split))
        for example in example_range:
            tokens = []
            valid_mask = []
            for word in example.words:
                word_tokens = self.tokenizer.tokenize(word)
                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                for i, word_token in enumerate(word_tokens):
                    if i == 0:
                        valid_mask.append(1)
                    else:
                        valid_mask.append(0)
                    tokens.append(word_token)
            label_ids = [label_map[label] for label in example.labels]
            entities = self.__get_entities(example.labels)
            start_ids = [span_map['O']] * len(label_ids)
            end_ids = [span_map['O']] * len(label_ids)
            for entity in entities:
                start_ids[entity[1]] = span_map[entity[0]]
                end_ids[entity[-1]] = span_map[entity[0]]
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]
                valid_mask = valid_mask[: (max_seq_length - special_tokens_count)]
                start_ids = start_ids[: (max_seq_length - special_tokens_count)]
                end_ids = end_ids[: (max_seq_length - special_tokens_count)]

            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            start_ids += [pad_token_label_id]
            end_ids += [pad_token_label_id]
            valid_mask.append(1)
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
                start_ids += [pad_token_label_id]
                end_ids += [pad_token_label_id]
                valid_mask.append(1)
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                start_ids += [pad_token_label_id]
                end_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
                valid_mask.append(1)
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                start_ids = [pad_token_label_id] + start_ids
                end_ids = [pad_token_label_id] + end_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                valid_mask.insert(0, 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
                start_ids = ([pad_token_label_id] * padding_length) + start_ids
                end_ids = ([pad_token_label_id] * padding_length) + end_ids
                valid_mask = ([0] * padding_length) + valid_mask
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length
                start_ids += [pad_token_label_id] * padding_length
                end_ids += [pad_token_label_id] * padding_length
                valid_mask += [0] * padding_length
            while (len(label_ids) < max_seq_length):
                label_ids.append(pad_token_label_id)
                start_ids.append(pad_token_label_id)
                end_ids.append(pad_token_label_id)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            try:
                assert len(label_ids) == max_seq_length
            except AssertionError:
                print(label_ids)
                print(len(label_ids), max_seq_length)
            assert len(start_ids) == max_seq_length
            assert len(end_ids) == max_seq_length
            assert len(valid_mask) == max_seq_length

            features.append(InputFeatures(input_ids=input_ids, input_mask=input_mask, valid_mask=valid_mask,
                                          segment_ids=segment_ids, label_ids=label_ids, start_ids=start_ids, end_ids=end_ids))
        return features


    def __end_of_chunk(self, prev_tag, tag, prev_type, type_):
        """Checks if a chunk ended between the previous and current word.
        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.
        Returns:
            chunk_end: boolean.
        """
        chunk_end = False
        if self.tag_scheme == 'IOB':
            if prev_tag == 'I' and tag in ['B', 'O']: chunk_end = True
            if prev_tag == 'I' and tag == 'I' and prev_type != type_: chunk_end = True
        if self.tag_scheme == 'IOB2':
            if prev_tag == 'I' and tag in ['B', 'O']: chunk_end = True
            if prev_tag == 'B' and tag == 'O': chunk_end = True
            if prev_tag == 'B' and tag == 'B' and prev_type != type_: chunk_end = True
        if self.tag_scheme == 'IOBES':
            if prev_tag in ['E', 'S']: chunk_end = True
        return chunk_end


    def __start_of_chunk(self, prev_tag, tag, prev_type, type_):
        """Checks if a chunk started between the previous and current word.
        Args:
            prev_tag: previous chunk tag.
            tag: current chunk tag.
            prev_type: previous type.
            type_: current type.
        Returns:
            chunk_start: boolean.
        """
        chunk_start = False
        if self.tag_scheme == 'IOB':
            if tag == 'B': chunk_start = True
            if prev_tag == 'O' and tag == 'I': chunk_start = True
            if prev_tag == 'I' and tag == 'I' and prev_type != type_: chunk_start = True
        if self.tag_scheme == 'IOB2':
            if tag == 'B': chunk_start = True
        if self.tag_scheme == 'IOBES':
            if tag in ['B', 'S']: chunk_start = True
        return chunk_start


    def __get_entities(self, seq):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (chunk_type, chunk_start, chunk_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-PER']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3], ['PER', 4, 4]]
        """
        if any(isinstance(s, list) for s in seq):
            seq = [item for sublist in seq for item in sublist + ['O']]
        prev_tag = 'O'
        prev_type = ''
        begin_offset = 0
        chunks = []
        for i, chunk in enumerate(seq + ['O']):
            tag = chunk[0]
            type_ = chunk.split('-')[-1]
            if self.__end_of_chunk(prev_tag, tag, prev_type, type_):
                chunks.append((prev_type, begin_offset, i - 1))
            if self.__start_of_chunk(prev_tag, tag, prev_type, type_):
                begin_offset = i
            prev_tag = tag
            prev_type = type_
        return set(chunks)


    def __collate_fn(self, batch):
        """
        batch should be a list of (sequence, target, length) tuples...
        Returns a padded tensor of sequences sorted from longest to shortest,
        """
        batch_tuple = tuple(map(torch.stack, zip(*batch)))
        batch_lens = torch.sum(batch_tuple[1], dim=-1, keepdim=False)
        max_len = batch_lens.max().item()
        results = ()
        for item in batch_tuple:
            if item.dim() >= 2:
                results += (item[:, :max_len],)
            else:
                results += (item,)
        return results


    def __get_classes(self, labels):
        classes_raw = labels
        classes = ["O"]
        if self.tag_scheme in ['IOB', 'IOB2']:
            prefixes = ['I', 'B']
        elif self.tag_scheme == 'IOBES':
            prefixes = ['B', 'I', 'E', 'S']
        classes.extend(['{}-{}'.format(p, c) for p in prefixes for c in classes_raw])
        self.classes = classes
        return classes


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, valid_mask, segment_ids, label_ids, start_ids, end_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.valid_mask = valid_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.start_ids = start_ids
        self.end_ids = end_ids


class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, words, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
