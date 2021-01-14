import json
import os
import numpy as np
import torch
from .bert_model import BertNER, BertCrfForNer
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm
from itertools import product
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup
from ..utils.data import create_tokenset


class NERModel:
    def __init__(self, model="allenai/scibert_scivocab_cased", labels=[], device="cpu", trained_ner=None):
        self.tokenizer = BertTokenizer.from_pretrained(model)
        self.labels = labels
        self.config = AutoConfig.from_pretrained(model)
        self.config.num_labels = 1 + 2*len(self.labels)
        self.device = device
        self.model = model
        self.trained_ner = trained_ner

    def load_file(self, data_file_path):
        data = []
        with open(data_file_path, 'r') as f:
            for l in f:
                data.append(json.loads(l))
        return data

    def preprocess(self, data):

        classes_raw = data[0]['labels']
        self.classes = ["O"]
        for c in classes_raw:
            self.classes.append("B-{}".format(c))
            self.classes.append("I-{}".format(c))

        data = [[(d['text'],d['annotation']) for d in s] for a in data for s in a['tokens']]

        input_examples = []
        max_sequence_length = 0
        for i, d in enumerate(data):
            labels = []
            text = []
            for t,l in d:

                #This causes issues with BERT for some reason
                if t in ['̄','̊']:
                    continue

                text.append(t)
                if l is None:
                    label = "O"
                elif "PUT" in l or "PVL" in l:
                    label = "O"
                else:
                    if len(labels) > 0 and l in labels[-1]:
                        label = "I-{}".format(l)
                    else:
                        label = "B-{}".format(l)
                labels.append(label)

            if len(text) > max_sequence_length:
                max_sequence_length = len(text)

            example = InputExample(i, text, labels)

            input_examples.append(example)

        features = self.__convert_examples_to_features(
                input_examples,
                self.classes,
                max_sequence_length,
        )

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids)

        return dataset

    def create_training_dataloaders(self, tensor_dataset, dev_frac, test_frac, batch_size, shuffle_dataset):
        dataset_size = len(tensor_dataset)
        indices = list(range(dataset_size))
        test_split = int(np.floor(test_frac * dataset_size))
        dev_split = int(np.floor(dev_frac * dataset_size))+test_split
        if shuffle_dataset:
            np.random.seed(1000)
            np.random.shuffle(indices)
        test_indices, dev_indices, train_indices = indices[:test_split], indices[test_split:dev_split], indices[test_split:]
        train_sampler = SubsetRandomSampler(train_indices)
        dev_sampler = SubsetRandomSampler(dev_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
            num_workers=0, sampler=train_sampler)
        dev_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
            num_workers=0, sampler=dev_sampler)
        test_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
            num_workers=0, sampler=test_sampler)
        return train_dataloader, dev_dataloader, test_dataloader

    def grid_search(self):
        return

    def accuracy(self, predicted, labels):
        predicted = torch.max(predicted, -1)[1]

        true = torch.where(labels > 0, labels, 0)
        predicted = torch.where(labels > 0, predicted, -1)

        acc = (true==predicted).sum().item()/torch.count_nonzero(true)
        return acc

    def train(self, data_file_path, dev_frac=0.02, test_frac=0.02, shuffle_dataset=True, lr=5e-5, n_epochs=10, batch_size=20):
        self.results_file = "ner_results.csv"

        self.data = self.load_file(data_file_path)

        print("{}-{}-{}".format(self.model.rsplit('/',1)[-1], lr, n_epochs))

        self.config.num_labels = 1 + 2 * max([len(datum['labels']) for datum in self.data])

        tensor_dataset = self.preprocess(self.data)
        train_dataloader, dev_dataloader, test_dataloader = self.create_training_dataloaders(tensor_dataset, dev_frac, test_frac, batch_size, shuffle_dataset)

        self.ner_model = BertCrfForNer(self.config).to(self.device)

        no_decay = ["bias", "LayerNorm.weight"]
        bert_parameters = self.ner_model.bert.named_parameters()
        classifier_parameters = self.ner_model.classifier.named_parameters()
        bert_lr = lr
        classifier_lr = lr
        optimizer_grouped_parameters = [
            {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": bert_lr},
            {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": bert_lr},

            {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": classifier_lr},
            {"params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0,
             "lr": classifier_lr}
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=n_epochs*len(train_dataloader)
        )

        self.eval_loss_best = 500000

        self.save_path = "{}_{}_{}_best.pt".format(self.model.rsplit('/',1)[-1], lr, n_epochs)
        for epoch in range(n_epochs):
            print("\n\n\nEpoch: " + str(epoch + 1))
            self.ner_model.train()

            for i, batch in enumerate(tqdm(train_dataloader)):
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device)
                }
                optimizer.zero_grad()
                loss, predicted = self.ner_model.forward(**inputs)
                loss.backward()
                optimizer.step()
                scheduler.step()

                if i%100 == 0:
                    labels = inputs['labels']
                    acc = self.accuracy(predicted, labels)

                    print("loss: {}, acc: {}".format(torch.mean(loss).item(), acc.item()))

            self.evaluate(dev_dataloader, validate=True)

        self.ner_model.load_state_dict(torch.load(self.save_path))
        self.evaluate(test_dataloader, validate=False, lr=lr, n_epochs=n_epochs)
        return

    def evaluate(self, dataloader, validate=False, lr=None, n_epochs=None):
        self.ner_model.eval()
        eval_loss = []
        eval_pred = []
        eval_label = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device)
                }
                loss, pred = self.ner_model.forward(**inputs)
                eval_loss.append(loss)
                eval_pred.append(pred)
                eval_label.append(inputs['labels'])
            if validate:
                eval_loss = torch.stack(eval_loss)
            else:
                eval_loss = torch.mean(torch.stack(eval_loss)).item()
            eval_pred = torch.cat(eval_pred, dim=0)
            eval_label = torch.cat(eval_label, dim=0)
            eval_acc = self.accuracy(eval_pred, eval_label)

        if validate:
            if torch.mean(eval_loss).item() < self.eval_loss_best:
                torch.save(self.ner_model.state_dict(), self.save_path)
            print("dev loss: {}, dev acc: {}".format(torch.mean(eval_loss).item(), eval_acc.item()))
        else:
            with open(self.results_file, "a+") as f:
                f.write("{},{},{},{},{}\n".format(self.model[0], lr, n_epochs, eval_loss, eval_acc.item()))

        return

    def predict(self, data):
        # check for input data type
        if os.path.isfile(data):
            texts = self.load_file(data)
        elif type(data) == list:
            texts = data
        elif type(data) == str:
            texts = [data]
        else:
            print("Please provide text or set of texts (directly or in a file path format) to predict on!")

        # tokenize and preprocess input data
        tokenized_dataset = []
        labels = self.labels
        for text in texts:
            tokenized_text = create_tokenset(text)
            tokenized_text['labels'] = labels
            tokenized_dataset.append(tokenized_text)
        tensor_dataset = self.preprocess(tokenized_dataset)
        pred_dataloader = DataLoader(tensor_dataset)
        self.ner_model = BertCrfForNer(self.config).to(self.device)
        self.ner_model.load_state_dict(torch.load(self.trained_ner))

        # run predictions
        with torch.no_grad():
            for i, batch in enumerate(pred_dataloader):
                # set up cursors for paragraphs and sentences in dataset since
                # some paragraphs have multiple sentences
                if i == 0:
                    para_i = 0
                    sent_i = 0
                    total_len = len(tokenized_dataset[para_i]['tokens'])
                elif i < total_len:
                    sent_i += 1
                else:
                    para_i += 1
                    sent_i = 0
                    total_len += len(tokenized_dataset[para_i]['tokens'])

                sentence = tokenized_dataset[para_i]['tokens'][sent_i]

                # get masked inputs and run predictions
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device)
                }
                loss, predicted = self.ner_model.forward(**inputs)
                predictions = torch.max(predicted,-1)[1]

                # assign predictions to dataset
                for tok in sentence:
                    try:
                        tok_idx = torch.tensor([sentence.index(tok)])
                        pred_idx = torch.index_select(predictions[:, 1:], 1, tok_idx)
                        tok['annotation'] = self.classes[pred_idx]
                    except:
                        print('reached max sequence length!')
                        continue

        return tokenized_dataset

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
        tokenizer = self.tokenizer
        label_map = {label: i for i, label in enumerate(label_list)}
        span_labels = []
        for label in label_list:
            label = label.split('-')[-1]
            if label not in span_labels:
                span_labels.append(label)
        span_map = {label: i for i, label in enumerate(span_labels)}
        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                print("Writing example %d of %d"%(ex_index, len(examples)))

            tokens = []
            valid_mask = []
            for word in example.words:
                word_tokens = tokenizer.tokenize(word)
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

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

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

        if prev_tag == 'E': chunk_end = True
        if prev_tag == 'S': chunk_end = True

        if prev_tag == 'B' and tag == 'B': chunk_end = True
        if prev_tag == 'B' and tag == 'S': chunk_end = True
        if prev_tag == 'B' and tag == 'O': chunk_end = True
        if prev_tag == 'I' and tag == 'B': chunk_end = True
        if prev_tag == 'I' and tag == 'S': chunk_end = True
        if prev_tag == 'I' and tag == 'O': chunk_end = True

        if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
            chunk_end = True

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

        if tag == 'B': chunk_start = True
        if tag == 'S': chunk_start = True

        if prev_tag == 'E' and tag == 'E': chunk_start = True
        if prev_tag == 'E' and tag == 'I': chunk_start = True
        if prev_tag == 'S' and tag == 'E': chunk_start = True
        if prev_tag == 'S' and tag == 'I': chunk_start = True
        if prev_tag == 'O' and tag == 'E': chunk_start = True
        if prev_tag == 'O' and tag == 'I': chunk_start = True

        if tag != 'O' and tag != '.' and prev_type != type_:
            chunk_start = True

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
