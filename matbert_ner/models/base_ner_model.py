import os
import json
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from utils.data import NERData
from itertools import product
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.metrics import accuracy
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.scheme import IOB1, IOB2, IOBES
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod

class NERModel(ABC):
    """
    A wrapper class for transformers models, implementing train, predict, and evaluate methods
    """

    def __init__(self, modelname="allenai/scibert_scivocab_cased", classes = ["O"], tag_format='IOB2', device="cpu", lr=5e-5, results_file=None):
        self.modelname = modelname
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.classes = classes
        self.tag_format = tag_format
        self.metric_mode = 'strict'
        if self.tag_format == 'IOB1':
            self.metric_scheme = IOB1
        elif self.tag_format == 'IOB2':
            self.metric_scheme = IOB2
        elif self.tag_format == 'IOBES':
            self.metric_scheme = IOBES
        self.config = AutoConfig.from_pretrained(modelname)
        self.config.num_labels = len(self.classes)
        self.config.model_name = self.modelname
        self.lr = lr
        self.device = device
        self.model = self.initialize_model()
        self.results_file = results_file


    def process_tags(self, inputs, predicted):
        labels = list(inputs['labels'].cpu().numpy())
        batch_size, max_len = inputs['input_ids'].shape
        valid_attention_mask = np.zeros((batch_size, max_len), dtype=int)
        for i in range(batch_size):
            jj = -1
            for j in range(max_len):
                if inputs['valid_mask'][i][j].item() == 1:
                    jj += 1
                    if inputs['input_ids'][i][j] not in (2, 3):
                        valid_attention_mask[i, jj] = inputs['attention_mask'][i][j].item()
        valid_attention_mask = list(valid_attention_mask)
        prediction_tags = [[self.classes[ii] for ii, jj in zip(i, j) if jj==1] for i, j in zip(predicted, valid_attention_mask)]
        label_tags = [[self.classes[ii] if ii>=0 else self.classes[0] for ii, jj in zip(i, j) if jj==1] for i, j in zip(labels, valid_attention_mask)]
        return prediction_tags, label_tags


    def train(self, train_dataloader, n_epochs, val_dataloader=None, dev_dataloader=None, save_dir=None, full_finetuning=True):
        """
        Train the model
        Inputs:
            dataloader :: dataloader with training data
            n_epochs :: number of epochs to train
            val_dataloader :: dataloader with validation data - if provided the model with the best performance on the validation set will be saved
            save_dir :: directory to save models
        """
        self.val_f1_best = 0

        optimizer = self.create_optimizer(full_finetuning)
        scheduler = self.create_scheduler(optimizer, n_epochs, train_dataloader)

        epoch_metrics = {'training': {}, 'validation': {}}
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)          

        for epoch in range(n_epochs):
            self.model.train()

            metrics = {'loss': [], 'accuracy_score': [], 'precision_score': [], 'recall_score': [], 'f1_score': []}
            batch_range = tqdm(train_dataloader, desc='')

            for i, batch in enumerate(batch_range):
                inputs = {"input_ids": batch[0].to(self.device, non_blocking=True),
                          "attention_mask": batch[1].to(self.device, non_blocking=True),
                          "valid_mask": batch[2].to(self.device, non_blocking=True),
                          "labels": batch[4].to(self.device, non_blocking=True)}

                optimizer.zero_grad()
                loss, predicted = self.model.forward(**inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                prediction_tags, label_tags = self.process_tags(inputs, predicted)

                metrics['loss'].append(loss.item())
                metrics['accuracy_score'].append(accuracy_score(label_tags, prediction_tags))
                metrics['precision_score'].append(precision_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['recall_score'].append(recall_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['f1_score'].append(f1_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]

                batch_range.set_description('| epoch: {:d}/{:d} | training | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1: {:.4f} |'.format(epoch+1, n_epochs, *means))
            
            if save_dir is not None:
                save_path = os.path.join(save_dir, "epoch_{}.pt".format(epoch))
                torch.save(self.model.state_dict(), save_path)

            epoch_metrics['training']['epoch_{}'.format(epoch)] = metrics

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader, validate=True, save_path=os.path.join(save_dir, "best.pt"))
                epoch_metrics['validation']['epoch_{}'.format(epoch)] = val_metrics

        if dev_dataloader is not None:
            # Restore weights of best model after training if we can
            save_path = os.path.join(save_dir, "best.pt")
            self.model.load_state_dict(torch.load(save_path))
            dev_metrics, dev_predictions, dev_labels = self.evaluate(dev_dataloader, validate=False)
            test_save_path = os.path.join(save_dir, 'test.pt')
            torch.save((dev_metrics, dev_predictions, dev_labels), test_save_path)
        
        history_save_path = os.path.join(save_dir, 'history.pt')
        torch.save(epoch_metrics, history_save_path)

        return

    def load_model(self,save_path):
        self.model.load_state_dict(torch.load(save_path))
        return

    def embed_documents(self, data):
        _, dataloader = self._data_to_dataloader(data)

        document_embeddings = []
        for i, batch in enumerate(tqdm(dataloader)):

                inputs = {"input_ids": batch[0].to(self.device, non_blocking=True),
                          "attention_mask": batch[1].to(self.device, non_blocking=True),}

                document_embedding = self.document_embeddings(**inputs)
                document_embeddings.append(document_embedding)

        return document_embeddings

    @abstractmethod
    def initialize_model(self):
        pass

    @abstractmethod
    def create_optimizer(self):
        pass

    @abstractmethod
    def create_scheduler(self, optimizer, n_epochs, train_dataloader):
        pass

    @abstractmethod
    def document_embeddings(self, **inputs):
        #Given an input dictionary, return the corresponding document embedding
        pass

    def evaluate(self, dataloader, validate=False, save_path=None, lr=None, n_epochs=None):
        self.model.eval()
        if validate:
            mode = 'validation'
        else:
            mode = 'test'
        if mode == 'test':
            prediction_tags_all = []
            label_tags_all = []
        metrics = {'loss': [], 'accuracy_score': [], 'precision_score': [], 'recall_score': [], 'f1_score': []}
        batch_range = tqdm(dataloader, desc='')
        with torch.no_grad():
            for batch in batch_range:
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device)
                }
                loss, predicted = self.model.forward(**inputs)

                prediction_tags, label_tags = self.process_tags(inputs, predicted)
                if mode == 'test':
                    prediction_tags_all.extend(prediction_tags)
                    label_tags_all.extend(label_tags)

                metrics['loss'].append(loss.item())
                metrics['accuracy_score'].append(accuracy_score(label_tags, prediction_tags))
                metrics['precision_score'].append(precision_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['recall_score'].append(recall_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['f1_score'].append(f1_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]

                batch_range.set_description('| {} | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1: {:.4f} |'.format(mode, *means))

        if validate:
            if means[4] > self.val_f1_best:
                torch.save(self.model.state_dict(), save_path)
                self.val_f1_best = means[4]
        elif self.results_file is not None:
            eval_loss
            with open(self.results_file, "a+") as f:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(self.model[0], lr, n_epochs, *means))

        if mode == 'test':
            return metrics, prediction_tags_all, label_tags_all
        else:
            return metrics

    def predict(self, data, trained_model=None, labels=None):
        self.model.eval()
        self.labels = labels

        if trained_model:
            self.model.load_state_dict(torch.load(trained_model))

        tokenized_dataset, pred_dataloader = self._data_to_dataloader(data)

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
                loss, predicted = self.model.forward(**inputs)
                predictions = predicted.to('cpu').numpy()[0]

                # assign predictions to dataset
                for tok in sentence:
                    tok_idx = torch.tensor([sentence.index(tok)])
                    pred_idx = predictions[tok_idx]
                    tok['annotation'] = self.classes[pred_idx]
                tokenized_dataset[para_i]['tokens'][sent_i] = sentence
        return tokenized_dataset

    def _data_to_dataloader(self, data):
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
        if self.labels:
            labels = self.labels
        else:
            labels = []
            for label in self.classes:
                if label != 'O' and label[2:] not in labels:
                    labels.append(label[2:])
        ner_data = NERData(modelname=self.modelname)
        tokenized_dataset = []
        for text in texts:
            tokenized_text = ner_data.create_tokenset(text)
            tokenized_text['labels'] = labels
            tokenized_dataset.append(tokenized_text)
        ner_data.preprocess(tokenized_dataset,is_file=False)
        tensor_dataset = ner_data.dataset
        pred_dataloader = DataLoader(tensor_dataset)

        return tokenized_dataset, pred_dataloader
