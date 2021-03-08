from transformers import BertTokenizer
from chemdataextractor.doc import Paragraph
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, SequentialSampler
import json
import torch
import numpy as np
from tqdm import tqdm

class NERData():

    def __init__(self, modelname="allenai/scibert_scivocab_cased", tag_format='IOB2'):
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.tag_format = tag_format
        self.dataset = None
        self.labels = None


    def load_from_file(self,datafile):
        data = []
        with open(datafile, 'r') as f:
            for l in f:
                data.append(json.loads(l))

        return data

    def preprocess(self, datafile, is_file=True):

        if is_file:
            data = self.load_from_file(datafile)
        else:
            data = datafile

        self.__get_tags(data[0]['labels'])

        texts = [[d['text'] for d in s] for a in data for s in a['tokens']]
        annotations = [[d['annotation'] for d in s] for a in data for s in a['tokens']]
        input_examples = []
        max_sequence_length = 0
        for n, (text, annotation) in enumerate(zip(texts, annotations)):
            txt = []
            label = []
            sequence_length = len(text)
            for i in range(sequence_length):
                if text[i] in ['̄','̊']:
                    continue
                txt.append(text[i])
                if self.tag_format == 'IOB':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        label.append('O')
                    elif i == 0:
                        if annotation[i+1] == annotation[i]:
                            label.append('B-'+annotation[i])
                        else:
                            label.append('I-'+annotation[i])
                    elif i > 0:
                        if annotation[i-1] == annotation[i]:
                            label.append('I-'+annotation[i])
                        else:
                            if annotation[i+1] == annotation[i]:
                                label.append('B-'+annotation[i])
                            else:
                                label.append('I-'+annotation[i])
                elif self.tag_format == 'IOB2':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        label.append('O')
                    elif i == 0:
                        label.append('B-'+annotation[i])
                    elif i > 0:
                        if annotation[i-1] == annotation[i]:
                            label.append('I-'+annotation[i])
                        else:
                            label.append('B-'+annotation[i])
                elif self.tag_format == 'IOBES':
                    if annotation[i] in [None, 'PVL', 'PUT']:
                        label.append('O')
                    elif i == 0:
                        if annotation[i+1] == annotation[i]:
                            label.append('B-'+annotation[i])
                        else:
                            label.append('S-'+annotation[i])
                    elif i > 0 and i < len(annotation)-1:
                        if annotation[i-1] != annotation[i] and annotation[i+1] == annotation[i]:
                            label.append('B-'+annotation[i])
                        elif annotation[i-1] == annotation[i] and annotation[i+1] == annotation[i]:
                            label.append('I-'+annotation[i])
                        elif annotation[i-1] == annotation[i] and annotation[i+1] != annotation[i]:
                            label.append('E-'+annotation[i])
                        if annotation[i-1] != annotation[i] and annotation[i+1] != annotation[i]:
                            label.append('S-'+annotation[i])
                    elif i == len(annotation)-1:
                        if annotation[i-1] == annotation[i]:
                            label.append('E-'+annotation[i])
                        if annotation[i-1] != annotation[i]:
                            label.append('S-'+annotation[i])
            sequence_length = len(txt)
            if sequence_length > max_sequence_length:
                max_sequence_length = sequence_length
            example = InputExample(n, txt, label)
            input_examples.append(example)
        
        features = self.__convert_examples_to_features(input_examples, self.classes, max_sequence_length)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids)

        self.dataset = dataset
        return self

    def create_dataloaders(self, batch_size=30, train_frac=None, val_frac=0.1, dev_frac=0.1, shuffle_dataset=True, seed=None):
        """
        Create train, val, and dev dataloaders from a preprocessed dataset
        Inputs:
            batch_size (int) :: Minibatch size for training
            train_frac (float or None) :: Fraction of data to use for training (None uses the remaining data)
            val_frac (float) :: Fraction of data to use for validation
            dev_frac (float) :: Fraction of data to use as a hold-out set
            shuffle_dataset (bool) :: Whether to randomize ordering of data samples
        Returns:
            dataloaders (tuple of torch.utils.data.Dataloader) :: train, val, and dev dataloaders
        """

        if self.dataset is None:
            print("No preprocessed dataset available")
            return None

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        dev_split = int(np.floor(dev_frac * dataset_size))
        val_split = int(np.floor(val_frac * dataset_size))+dev_split
        if shuffle_dataset:
            np.random.seed(seed)
            np.random.shuffle(indices)

        dev_indices, val_indices = indices[:dev_split], indices[dev_split:val_split]

        if train_frac:
            train_split = int(np.floor(train_frac * dataset_size))+val_split
            train_indices = indices[val_split:train_split]
        else:
             train_indices = indices[val_split:]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SequentialSampler(val_indices)
        dev_sampler = SequentialSampler(dev_indices)

        self.train_dataloader = DataLoader(self.dataset, batch_size=batch_size,
            num_workers=0, sampler=train_sampler, pin_memory=True)

        if val_frac > 0:
            self.val_dataloader = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=0, sampler=val_sampler, pin_memory=True)
        else:
            self.val_dataloader = None

        if dev_frac > 0:
            self.dev_dataloader = DataLoader(self.dataset, batch_size=batch_size,
                num_workers=0, sampler=dev_sampler, pin_memory=True)
        else:
            self.dev_dataloader = None

        return self.train_dataloader, self.val_dataloader, self.dev_dataloader

    def create_tokenset(self, text):

        tokenset = {
            "text" : text,
            "tokens" : []
        }

        idx = 0

        para = Paragraph(text)
        sentences = para.raw_sentences

        for sentence in sentences:
            tokens = self.tokenizer.tokenize(sentence)
            sent_toks = []
            for token in tokens:
                if token.startswith('##'):
                    tok_length = len(token) - 2
                else:
                    tok_length = len(token)
                tok = {
                    "text" : token,
                    "start" : idx,
                    "end" : idx + tok_length,
                    "annotation" : None
                }

                if idx + tok_length >= len(sentence):
                    sent_toks.append(tok)
                    break
                elif sentence[idx + tok_length] == " ":
                    idx += tok_length + 1
                else:
                    idx += tok_length
                sent_toks.append(tok)

            tokenset["tokens"].append(sent_toks)

        return tokenset

    def __convert_examples_to_features(
            self,
            examples,
            label_list,
            max_seq_length,
            cls_token_at_end=False,
            cls_token="[CLS]",
            cls_token_segment_id=1,
            sep_token="[SEP]",
            sep_token_extra=False,
            pad_on_left=False,
            pad_token=0,
            pad_token_segment_id=0,
            pad_token_label_id=-100,
            sequence_a_segment_id=0,
            mask_padding_with_zero=True,
    ):
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
        example_range = tqdm(examples, desc='| writing examples |')
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

            features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              valid_mask=valid_mask,
                              segment_ids=segment_ids,
                              label_ids=label_ids,
                              start_ids=start_ids,
                              end_ids=end_ids)
            )
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
        if self.tag_format == 'IOB':
            if prev_tag == 'I' and tag in ['B', 'O']: chunk_end = True
            if prev_tag == 'I' and tag == 'I' and prev_type != type_: chunk_end = True
        if self.tag_format == 'IOB2':
            if prev_tag == 'I' and tag in ['B', 'O']: chunk_end = True
            if prev_tag == 'B' and tag == 'O': chunk_end = True
            if prev_tag == 'B' and tag == 'B' and prev_type != type_: chunk_end = True
        if self.tag_format == 'IOBES':
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
        if self.tag_format == 'IOB':
            if tag == 'B': chunk_start = True
            if prev_tag == 'O' and tag == 'I': chunk_start = True
            if prev_tag == 'I' and tag == 'I' and prev_type != type_: chunk_start = True
        if self.tag_format == 'IOB2':
            if tag == 'B': chunk_start = True
        if self.tag_format == 'IOBES':
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

    def __get_tags(self, labels):
        classes_raw = labels
        classes = ["O"]
        if self.tag_format in ['IOB', 'IOB2']:
            prefixes = ['I', 'B']
        elif self.tag_format == 'IOBES':
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
