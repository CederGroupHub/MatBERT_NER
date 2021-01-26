import os
import json
import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SubsetRandomSampler
from utils.data import create_tokenset, NERData
from itertools import product
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup
from tqdm import tqdm
from utils.metrics import accuracy
from seqeval.metrics import accuracy_score, f1_score
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod

class NERModel(ABC):
    """
    A wrapper class for transformers models, implementing train, predict, and evaluate methods
    """

    def __init__(self, modelname="allenai/scibert_scivocab_cased", classes = ["O"], device="cpu", lr=5e-5, results_file=None):
        self.modelname = modelname
        self.tokenizer = BertTokenizer.from_pretrained(modelname)
        self.classes = classes
        self.config = AutoConfig.from_pretrained(modelname)
        self.config.num_labels = len(self.classes)
        self.lr = lr
        self.device = device
        self.model = self.initialize_model()
        self.results_file = results_file


    def train(self, train_dataloader, n_epochs, val_dataloader=None, save_dir=None):
        """
        Train the model
        Inputs:
            dataloader :: dataloader with training data
            n_epochs :: number of epochs to train
            val_dataloader :: dataloader with validation data - if provided the model with the best performance on the validation set will be saved
            save_dir :: directory to save models
        """

        self.val_loss_best = 1e10


        optimizer = self.create_optimizer()
        scheduler = self.create_scheduler(optimizer, n_epochs, train_dataloader)

        epoch_metrics = {'train': {}, 'validate': {}}

        for epoch in range(n_epochs):
            self.model.train()

            metrics = {'loss': [], 'accuracy': [],  'accuracy_score': [], 'f1_score': []}
            batch_range = tqdm(train_dataloader, desc='')

            for i, batch in enumerate(batch_range):
                
                inputs = {"input_ids": batch[0].to(self.device, non_blocking=True),
                          "attention_mask": batch[1].to(self.device, non_blocking=True),
                          "valid_mask": batch[2].to(self.device, non_blocking=True),
                          "labels": batch[4].to(self.device, non_blocking=True)}

                optimizer.zero_grad()
                loss, predicted = self.model.forward(**inputs)
                loss.backward()
                optimizer.step()
                scheduler.step()

                labels = inputs['labels']
                labels_list = list(labels.cpu().numpy())

                prediction = torch.max(predicted,-1)[1]
                prediction_list = list(prediction.cpu().numpy())

                prediction_tags = [[self.classes[ii] for ii, jj in zip(i, j) if jj != -100] for i, j in zip(prediction_list, labels_list)]
                valid_tags = [[self.classes[ii] for ii in i if ii != -100] for i in labels_list]
                
                metrics['loss'].append(loss.item())
                metrics['accuracy'].append(accuracy(predicted, labels).item())
                metrics['accuracy_score'].append(accuracy_score(valid_tags, prediction_tags))
                metrics['f1_score'].append(f1_score(valid_tags, prediction_tags))
                # metric_list = ['loss', 'accuracy']
                metric_list = ['loss', 'accuracy', 'accuracy_score', 'f1_score']
                means = [np.mean(metrics[metric]) for metric in metric_list]

                batch_range.set_description('| training | epoch: {:d}/{:d} | loss: {:.4f} | accuracy: {:.4f} | accuracy score: {:.4f} | f1 score: {:.4f} |'.format(epoch+1, n_epochs, *means))
                # batch_range.set_description('| epoch: {:d}/{:d} | loss: {:.4f} | accuracy: {:.4f} |'.format(epoch+1, n_epochs, *means))

            save_path = os.path.join(save_dir, "epoch_{}.pt".format(epoch))
            torch.save(self.model.state_dict(), save_path)

            epoch_metrics['train']['epoch_{}'.format(epoch)] = metrics

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader, validate=True, save_path=os.path.join(save_dir, "best.pt"))
                epoch_metrics['validate']['epoch_{}'.format(epoch)] = val_metrics
        if val_dataloader is not None:
            # Restore weights of best model after training if we can

            save_path = os.path.join(save_dir, "best.pt")
            self.model.load_state_dict(torch.load(save_path))
            self.evaluate(val_dataloader, validate=False)
        
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
        eval_loss = []
        eval_pred = []
        eval_label = []
        prediction_tags_all = []
        valid_tags_all = []
        metrics = {'loss': [], 'accuracy': [],  'accuracy_score': [], 'f1_score': []}
        with torch.no_grad():
            for batch in dataloader:
                inputs = {
                    "input_ids": batch[0].to(self.device),
                    "attention_mask": batch[1].to(self.device),
                    "valid_mask": batch[2].to(self.device),
                    "labels": batch[4].to(self.device)
                }
                loss, predicted = self.model.forward(**inputs)
                labels = inputs['labels']

                labels_list = list(labels.cpu().numpy())
                prediction_list = list(torch.max(predicted,-1)[1].cpu().numpy())

                eval_loss.append(loss)
                eval_pred.append(predicted)
                eval_label.append(labels)

                prediction_tags = [[self.classes[ii] for ii, jj in zip(i, j) if jj != -100] for i, j in zip(prediction_list, labels_list)]
                valid_tags = [[self.classes[ii] for ii in i if ii != -100] for i in labels_list]

                prediction_tags_all.extend(prediction_tags)
                valid_tags_all.extend(valid_tags)

                metrics['loss'].append(loss.item())
                metrics['accuracy'].append(accuracy(predicted, labels).item())
                metrics['accuracy_score'].append(accuracy_score(valid_tags, prediction_tags))
                metrics['f1_score'].append(f1_score(valid_tags, prediction_tags))

            eval_loss = torch.mean(torch.stack(eval_loss)).item()
            eval_pred = torch.cat(eval_pred, dim=0)
            eval_label = torch.cat(eval_label, dim=0)
            eval_acc = accuracy(eval_pred, eval_label).item()
            eval_acc_score = accuracy_score(valid_tags_all, prediction_tags_all)
            eval_f1_score = f1_score(valid_tags_all, prediction_tags_all)

        if validate:
            if eval_loss < self.val_loss_best:
                torch.save(self.model.state_dict(), save_path)
                self.val_loss_best = eval_loss
            mode = 'validation'
        elif self.results_file is not None:
            with open(self.results_file, "a+") as f:
                f.write("{},{},{},{},{},{},{}\n".format(self.model[0], lr, n_epochs, eval_loss, eval_acc, eval_acc_score, eval_f1_score))
            mode = 'test'
        else:
            mode = 'test'

        print("| {} | loss: {:.4f} | accuracy: {:.4f} | accuracy_score: {:.4f} | f1_score: {:.4f} |".format(mode, eval_loss, eval_acc, eval_acc_score, eval_f1_score))

        if validate:
            return metrics
        else:
            return

    def predict(self, data):

        tokenized_text, pred_dataloader = self._data_to_dataloader(data)

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
                    "labels": batch[4].to(self.device),
                    "decode": True
                }
                loss, predicted = self.model.forward(**inputs)
                predictions = torch.max(predicted,-1)[1]

                # assign predictions to dataset
                for tok in sentence:
                    try:
                        tok_idx = torch.tensor([sentence.index(tok)])
                        pred_idx = torch.index_select(predictions[:, 1:], 1, tok_idx)
                        tok['annotation'] = self.classes[pred_idx]
                    except ValueError:
                        print('reached max sequence length!')
                        continue

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
        tokenized_dataset = []
        labels = self.classes
        for text in texts:
            tokenized_text = create_tokenset(text)
            tokenized_text['labels'] = labels
            tokenized_dataset.append(tokenized_text)
        ner_data = NERData(modelname=self.modelname)
        ner_data.classes = labels
        ner_data.preprocess(tokenized_dataset,is_file=False)
        tensor_dataset = ner_data.dataset
        pred_dataloader = DataLoader(tensor_dataset)

        return tokenized_text, pred_dataloader