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
from pprint import pprint

class NERModel(ABC):
    """
    A wrapper class for transformers models, implementing train, predict, and evaluate methods
    """

    def __init__(self, modelname="allenai/scibert_scivocab_cased", classes = ["O"], tag_scheme='IOB2', device="cpu", elr=1e-2, tlr=1e-2, clr=1e-2, seed=256, results_file=None):
        self.modelname = modelname
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.classes = classes
        self.tag_scheme = tag_scheme
        self.metric_mode = 'strict'
        if self.tag_scheme == 'IOB1':
            self.metric_scheme = IOB1
        elif self.tag_scheme == 'IOB2':
            self.metric_scheme = IOB2
        elif self.tag_scheme == 'IOBES':
            self.metric_scheme = IOBES
        self.config = AutoConfig.from_pretrained(modelname)
        self.config.num_labels = len(self.classes)
        self.config.model_name = self.modelname
        self.elr = elr
        self.tlr = tlr
        self.clr = clr
        self.seed = seed
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
        return label_tags, prediction_tags


    def train(self, n_epochs, train_dataloader, val_dataloader=None, dev_dataloader=None, save_dir=None, opt_name='adamw', embedding_unfreeze=0, encoder_schedule=[12]):
        """
        Train the model
        Inputs:
            dataloader :: dataloader with training data
            n_epochs :: number of epochs to train
            val_dataloader :: dataloader with validation data - if provided the model with the best performance on the validation set will be saved
            save_dir :: directory to save models
        """
        self.val_f1_best = -1
        n_batches = len(train_dataloader)

        optimizer = self.create_optimizer(opt_name)
        scheduler = self.create_scheduler(optimizer, n_epochs)

        epoch_metrics = {'training': {}, 'validation': {}}
        if save_dir is not None and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.model.bert.encoder.parameters():
            param.requires_grad = False
        print('BERT embeddings and encoders frozen')
        print('CRF, Classifier, and BERT pooler unfrozen')
        
        last_encoder_layer = 11
        expanded_encoder_schedule = {}
        for epoch in range(n_epochs):
            expanded_encoder_schedule['epoch_{}'.format(epoch)] = []
            for layer in range(encoder_schedule[epoch]):
                expanded_encoder_schedule['epoch_{}'.format(epoch)].append(last_encoder_layer)
                last_encoder_layer -= 1


        for epoch in range(n_epochs):
            self.model.train()

            metrics = {'loss': [], 'accuracy_score': [], 'precision_score': [], 'recall_score': [], 'f1_score': []}
            batch_range = tqdm(train_dataloader, desc='')
            
            for layer_index in expanded_encoder_schedule['epoch_{}'.format(epoch)]:
                for param in self.model.bert.encoder.layer[layer_index].parameters():
                    param.requires_grad = True
                print('BERT encoder {} unfrozen'.format(layer_index))

            if epoch == embedding_unfreeze:
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = True
                print('BERT embeddings unfrozen')

            for j, batch in enumerate(batch_range):
                inputs = {"input_ids": batch[0].to(self.device, non_blocking=True),
                          "attention_mask": batch[1].to(self.device, non_blocking=True),
                          "valid_mask": batch[2].to(self.device, non_blocking=True),
                          "labels": batch[4].to(self.device, non_blocking=True)}

                optimizer.zero_grad()
                loss, predicted = self.model.forward(**inputs)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=1.0)
                optimizer.step()

                label_tags, prediction_tags = self.process_tags(inputs, predicted)

                metrics['loss'].append(loss.item())
                metrics['accuracy_score'].append(accuracy_score(label_tags, prediction_tags))
                metrics['precision_score'].append(precision_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['recall_score'].append(recall_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['f1_score'].append(f1_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]

                batch_range.set_description('| epoch: {:d}/{:d} | train | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1: {:.4f} |'.format(epoch+1, n_epochs, *means))
            
            if save_dir is not None:
                save_path = os.path.join(save_dir, "epoch_{}.pt".format(epoch))
                torch.save(self.model.state_dict(), save_path)

            epoch_metrics['training']['epoch_{}'.format(epoch)] = metrics

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader, validate=True, save_path=os.path.join(save_dir, "best.pt"), epoch=epoch, n_epochs=n_epochs)
                epoch_metrics['validation']['epoch_{}'.format(epoch)] = val_metrics
            
            scheduler.step()

        if dev_dataloader is not None:
            # Restore weights of best model after training if we can
            save_path = os.path.join(save_dir, "best.pt")
            self.model.load_state_dict(torch.load(save_path))
            dev_metrics, dev_text, dev_attention, dev_valid, dev_label, dev_prediction = self.evaluate(dev_dataloader, validate=False)
            test_save_path = os.path.join(save_dir, 'test.pt')
            torch.save((dev_metrics, dev_text, dev_attention, dev_valid, dev_label, dev_prediction), test_save_path)
        
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

    def evaluate(self, dataloader, validate=False, save_path=None, epoch=0, n_epochs=1):
        self.model.eval()
        if validate:
            mode = 'valid'
        else:
            mode = 'test'
        if mode == 'test':
            tokens_all = []
            attention_all = []
            valid_all = []
            label_all = []
            prediction_all = []
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

                label_tags, prediction_tags = self.process_tags(inputs, predicted)
                if mode == 'test':
                    tokens_all.extend(list(inputs['input_ids'].cpu().numpy()))
                    attention_all.extend(list(inputs['attention_mask'].cpu().numpy()))
                    valid_all.extend(list(inputs['valid_mask'].cpu().numpy()))
                    label_all.extend(label_tags)
                    prediction_all.extend(prediction_tags)

                metrics['loss'].append(loss.item())
                metrics['accuracy_score'].append(accuracy_score(label_tags, prediction_tags))
                metrics['precision_score'].append(precision_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['recall_score'].append(recall_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                metrics['f1_score'].append(f1_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme))
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]

                batch_range.set_description('| epoch: {:d}/{:d} | {} | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1: {:.4f} |'.format(epoch+1, n_epochs, mode, *means))

        if validate:
            if means[4] >= self.val_f1_best:
                torch.save(self.model.state_dict(), save_path)
                self.val_f1_best = means[4]
        elif self.results_file is not None:
            eval_loss
            with open(self.results_file, "a+") as f:
                f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(self.model[0], lr, n_epochs, *means))

        if mode == 'test':
            return metrics, tokens_all, attention_all, valid_all, label_all, prediction_all
        else:
            return metrics

    def predict(self, data, labels=None, trained_model=None, tok_dataset=None, return_tags=False, threshold=None):
        """
            Method for predicting NER labels based on trained model
            input: data to be predicted (single string, list of strings, or preprocessed dataloader objects),
            labels for entities (optional), trained model (optional), tokenized_dataset (optional, needed if loading
            preprocessed dataloader).
            return_tags :: whether to return prediction and label tag tensors
            threshold :: Threshold for classification of an entity. If None, Viterbi decoding is used
            returns: token set with predicted labels
        """

        if labels:
            self.labels = labels
        else:
            self.labels = []

        if type(data) == torch.utils.data.dataloader.DataLoader:
            pred_dataloader = data
            tokenized_dataset = tok_dataset

        else:
            if type(data) == str:
                data = [data]

            ner_data = NERData(self.modelname)

            tokenized_dataset = []
            for para in data:
                token_set = ner_data.create_tokenset(para)
                token_set['labels'] = self.labels
                tokenized_dataset.append(token_set)

            ner_data.preprocess(tokenized_dataset, is_file=False)
            tensor_dataset = ner_data.dataset
            pred_dataloader = DataLoader(tensor_dataset)

            #self.classes = ner_data.classes
            self.config.num_labels = len(self.classes)
            # self.model = self.initialize_model()

        if trained_model:
            try:
                self.model.load_state_dict(torch.load(trained_model))
            except:
                self.model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))

        # run predictions
        all_prediction_tags = []
        all_label_tags = []
        all_losses = []
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
                    "labels": batch[4].to(self.device),
                    "return_logits": True
                }

                loss, predicted, logits = self.model.forward(**inputs)

                if return_tags:
                    label_tags, prediction_tags = self.process_tags(inputs, predicted)
                    all_prediction_tags.append(prediction_tags)
                    all_label_tags.append(label_tags)
                    all_losses.append(loss)
                # predictions = torch.max(predicted, -1)[1]
                if threshold is None:
                    predicted = predicted[0][1:-1]
                else:
                    m = torch.nn.Softmax(dim=-1)
                    preds = m(logits)
                    max_pred = torch.max(preds, dim=-1)
                    max_pred_values = max_pred[0]
                    max_pred_indices = max_pred[1]
                    above_threshold_mask = (max_pred_values > threshold)
                    predicted = torch.where(above_threshold_mask, max_pred_indices, torch.zeros_like(max_pred_indices))
                    predicted = predicted.tolist()[0][1:-1] 

                # assign predictions to dataset
                #print(predicted)
                #print(sentence)
                #print(len(predicted))
                #print(len(sentence))
                for i, pred_idx in enumerate(predicted):
                    if i < len(sentence)-1:
                        tok = sentence[i]
                        #pred_idx = predicted[i]
                        tok['annotation'] = self.classes[pred_idx]
                tokenized_dataset[para_i]['tokens'][sent_i] = sentence
        if return_tags:
            return tokenized_dataset, [x for k in all_prediction_tags for x in k], [x for k in all_label_tags for x in k], torch.stack(all_losses)
        else:
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
