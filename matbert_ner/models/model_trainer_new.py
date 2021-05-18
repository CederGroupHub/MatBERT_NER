import copy
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW
from torchtools.optim import RangerLars
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score


class StateCacher(object):
    def __init__(self):
        self.cached = {}


    def store(self, key, state_dict):
        self.cached.update({key: copy.deepcopy(state_dict)})


    def retrieve(self, key):
        if key not in self.cached:
            raise KeyError('target {} was not cached.'.format(key))
        return self.cached.get(key)


class NERTrainer(object):
    def __init__(self, model, tokenizer, device):
        '''
        class for basic functions common to the trainer objects used in this project

        model: the model to be trained
        device: torch device
        '''
        self.max_grad_norm = 1.0
        self.device = device
        # send model to device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.state_cacher = StateCacher()
        self.optimizer = None
        self.scheduler = None
        self.metric_mode = 'strict'
        if self.model.tag_scheme == 'IOB1':
            self.metric_scheme = IOB1
        elif self.model.tag_scheme == 'IOB2':
            self.metric_scheme = IOB2
        elif self.model.tag_scheme == 'IOBES':
            self.metric_scheme = IOBES
        self.past_epoch = 0
    

    def save_model(self, model_path):
        ''' saves entire model to file '''
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_name': self.optimizer_name,
                    'learning_rates': {'elr': self.elr, 'tlr': self.tlr, 'clr': self.clr},
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'n_epoch': self.n_epoch,
                    'bert_unfreeze': self.bert_unfreeze,
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'classes': self.model.classes}, model_path)
    

    def load_model(self, model_path):
        ''' loads entire model from file '''
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.init_optimizer(checkpoint['optimizer_name'], **checkpoint['learning_rates'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.init_scheduler(checkpoint['n_epoch']-checkpoint['bert_unfreeze'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.model.classes = checkpoint['classes']
        self.past_epoch = checkpoint['n_epoch']


    def save_state_to_cache(self, key):
        self.state_cacher.store('model_state_dict_{}'.format(key), self.model.state_dict())
        self.state_cacher.store('optimizer_name_{}'.format(key), self.optimizer_name)
        self.state_cacher.store('learning_rates_{}'.format(key), {'elr': self.elr, 'tlr': self.tlr, 'clr': self.clr})
        self.state_cacher.store('optimizer_state_dict_{}'.format(key), self.optimizer.state_dict())
        self.state_cacher.store('n_epoch_{}'.format(key), self.n_epoch)
        self.state_cacher.store('bert_unfreeze_{}'.format(key), self.bert_unfreeze)
        self.state_cacher.store('scheduler_state_dict_{}'.format(key), self.scheduler.state_dict())
        self.state_cacher.store('classes_{}'.format(key), self.model.classes)
    

    def load_state_from_cache(self, key):
        self.model.load_state_dict(self.state_cacher.retrieve('model_state_dict_{}'.format(key)))
        self.model = self.model.to(self.device)
        self.init_optimizer(self.state_cacher.retrieve('optimizer_name_{}'.format(key)),
                            **self.state_cacher.retrieve('learning_rates_{}'.format(key)))
        self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer_state_dict_{}'.format(key)))
        self.init_scheduler(self.state_cacher.retrieve('n_epoch_{}'.format(key))-self.state_cacher.retrieve('bert_unfreeze_{}'.format(key)))
        self.scheduler.load_state_dict(self.state_cacher.retrieve('scheduler_state_dict_{}'.format(key)))
        self.model.classes = self.state_cacher.retrieve('classes_{}'.format(key))
    

    def save_history(self, history_path):
        ''' save training histories to file '''
        torch.save(self.epoch_metrics, history_path)
    

    def load_history(self, history_path):
        ''' load training histories from file '''
        self.epoch_metrics = torch.load(history_path)
        self.past_epoch = len(self.epoch_metrics['training'])


    def get_history(self):
        ''' get history '''
        return self.epoch_metrics
    

    def init_optimizer(self, name, elr, tlr, clr):
        if name == 'adamw':
            self.optimizer = AdamW([{'params': self.model.bert.embeddings.parameters(), 'lr': elr},
                                    {'params': self.model.bert.encoder.parameters(), 'lr': tlr},
                                    {'params': self.model.bert.pooler.parameters(), 'lr': clr},
                                    {'params': self.model.classifier.parameters(), 'lr': clr},
                                    {'params': self.model.crf.parameters(), 'lr': clr}])
        if name == 'rangerlars':
            self.optimizer = RangerLars([{'params': self.model.bert.embeddings.parameters(), 'lr': elr},
                                         {'params': self.model.bert.encoder.parameters(), 'lr': tlr},
                                         {'params': self.model.bert.pooler.parameters(), 'lr': clr},
                                         {'params': self.model.classifier.parameters(), 'lr': clr},
                                         {'params': self.model.crf.parameters(), 'lr': clr}])
    

    def init_scheduler(self, n_epoch):
        linear = lambda epoch: (n_epoch-epoch)/(n_epoch)
        exponential = lambda epoch: 0.01**(epoch/(n_epoch-1))
        cosine = lambda epoch: 0.5*(1+np.cos(epoch/n_epoch*np.pi))
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=linear, verbose=True)
    

    def process_tags(self, inputs, predicted, mode):
        if mode != 'predict':
            batch_size, max_len = inputs['input_ids'].shape
            labels = list(inputs['labels'].cpu().numpy())
            valid_attention_mask = np.zeros((batch_size, max_len), dtype=int)

            for i in range(batch_size):
                jj = -1
                for j in range(max_len):
                    if inputs['valid_mask'][i][j].item() == 1:
                        jj += 1
                        if inputs['input_ids'][i][j] not in (2, 3):
                            valid_attention_mask[i, jj] = inputs['attention_mask'][i][j].item()

            valid_attention_mask = list(valid_attention_mask)
            prediction_tags = [[self.model.classes[ii] for ii, jj in zip(i, j) if jj==1] for i, j in zip(predicted, valid_attention_mask)]
            label_tags = [[self.model.classes[ii] if ii>=0 else self.model.classes[0] for ii, jj in zip(i, j) if jj==1] for i, j in zip(labels, valid_attention_mask)]

            return label_tags, prediction_tags
        else:
            prediction_tags = [[self.model.classes[j] for j in i] for i in predicted]
            return prediction_tags
    

    def iterate_batches(self, epoch, n_epoch, iterator, mode):
        '''
        iterates through batchs in an epoch

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        mode: train, evaluate, or test
        '''
        # initialize lists for batch losses and metrics
        if mode != 'predict':
            metrics = {'loss': [], 'accuracy_score': [], 'precision_score': [], 'recall_score': [], 'f1_score': []}
        if mode == 'test' or mode == 'predict':
            tokens_all = []
            attention_all = []
            valid_all = []
            prediction_all = []
        if mode == 'test':
            label_all = []
        if mode == 'predict':
            logits_all = []
        # initialize batch range
        batch_range = tqdm(iterator, desc='')
        for batch in batch_range:
            # fetch texts, characters, and tags from batch
            inputs = {'input_ids': batch[0].to(self.device, non_blocking=True),
                      'attention_mask': batch[1].to(self.device, non_blocking=True),
                      'valid_mask': batch[2].to(self.device, non_blocking=True),
                      'device': self.device}
            if mode != 'predict':
                inputs['labels'] = batch[4].to(self.device, non_blocking=True)
            if mode == 'predict':
                inputs['return_logits'] = True

            # zero out prior gradients for training
            if mode == 'train':
                self.optimizer.zero_grad()

            # output depends on whether conditional random field is used for prediction/loss
            if mode != 'predict':
                loss, predicted = self.model.forward(**inputs)
                label_tags, prediction_tags = self.process_tags(inputs, predicted, mode)
            else:
                predicted, logits = self.model.forward(**inputs)
                prediction_tags = self.process_tags(inputs, predicted, mode)

            if mode == 'test' or mode == 'predict':
                tokens_all.extend(tokenizer.batch_decode(inputs['input_ids']))
                attention_all.extend(list(inputs['attention_mask'].cpu().numpy()))
                valid_all.extend(list(inputs['valid_mask'].cpu().numpy()))
                prediction_all.extend(prediction_tags)
            if mode == 'test':
                label_all.extend(label_tags)
            if mode == 'predict':
                logits_all.extend(list(logits.cpu().numpy()))

            if mode != 'predict':
                # calculate the accuracy and f1 scores
                accuracy = accuracy_score(label_tags, prediction_tags)
                precision = precision_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme)
                recall = recall_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme)
                f1 = f1_score(label_tags, prediction_tags, mode=self.metric_mode, scheme=self.metric_scheme)

                # append to the lists
                metrics['loss'].append(loss.item())
                metrics['accuracy_score'].append(accuracy)
                metrics['precision_score'].append(precision)
                metrics['recall_score'].append(recall)
                metrics['f1_score'].append(f1)
                # calculate means across the batches so far
                means = [np.mean(metrics[metric]) for metric in metrics.keys()]

            # backpropagate the gradients and step the optimizer forward
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            if mode != 'predict':
                # display progress
                msg = '| epoch: {:d}/{:d} | {} | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1: {:.4f} |'
                info = (self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode, *means)
            else:
                msg = ''
                info = ()
            batch_range.set_description(msg.format(*info))
        # return the batch losses and metrics
        if mode == 'test':
            return metrics, tokens_all, attention_all, valid_all, label_all, prediction_all
        elif mode != 'predict':
            return metrics
        else:
            return tokens_all, attention_all, valid_all, prediction_all, logits_all
    

    def train_evaluate_epoch(self, epoch, n_epoch, iterator, mode):
        '''
        train or evaluate epoch (calls the iterate_batches method from a subclass that inherits from this class)

        epoch: current epoch
        n_epoch: total epochs
        iterator: the iterator to be used for fetching batches
        mode: train, evaluate, or test
        '''
        if mode == 'train':
            # make sure the model is set to train if it is training
            self.model.train()
            # train all of the batches and collect the batch/epoch loss/metrics
            metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
        elif mode == 'valid':
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
        elif mode == 'test':
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                metrics, tokens, attention, valid, label, prediction = self.iterate_batches(epoch, n_epoch, iterator, mode)
        elif mode == 'predict':
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                tokens, attention, valid, prediction, logits = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # return batch/epoch loss/metrics
        if mode == 'test':
            return metrics, tokens, attention, valid, label, prediction
        elif mode != 'predict':
            return metrics
        else:
            return tokens, attention, valid, prediction, logits
    

    def train(self, n_epoch, train_iter, valid_iter, optimizer_name, elr, tlr, clr, embedding_unfreeze, encoder_schedule):
        '''
        trains the model (with validation)

        n_epoch: number of training epochs
        '''
        self.epoch_metrics = {'training': {}}
        if valid_iter:
            self.epoch_metrics['validation'] = {}

        encoder_unfreeze = next((i for i, n in enumerate(encoder_schedule) if n), n_epoch)
        bert_unfreeze = encoder_unfreeze if encoder_unfreeze < embedding_unfreeze else embedding_unfreeze

        self.optimizer_name = optimizer_name
        self.elr, self.tlr, self.clr = elr, tlr, clr
        self.init_optimizer(optimizer_name, elr, tlr, clr)
        self.n_epoch, self.embedding_unfreeze = n_epoch, embedding_unfreeze
        self.init_scheduler(n_epoch-bert_unfreeze)

        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.model.bert.encoder.parameters():
            param.requires_grad = False
        print('BERT embeddings and encoders frozen')
        print('CRF, Classifier, and BERT pooler unfrozen')

        last_encoder_layer = 11
        expanded_encoder_schedule = {}
        for epoch in range(n_epoch):
            expanded_encoder_schedule['epoch_{}'.format(epoch)] = []
            for layer in range(encoder_schedule[epoch]):
                expanded_encoder_schedule['epoch_{}'.format(epoch)].append(last_encoder_layer)
                last_encoder_layer -= 1

        best_validation_f1 = 0.0

        for epoch in range(n_epoch):
            for layer_index in expanded_encoder_schedule['epoch_{}'.format(epoch)]:
                for param in self.model.bert.encoder.layer[layer_index].parameters():
                    param.requires_grad = True
                print('BERT encoder {} unfrozen'.format(layer_index))
            if epoch == embedding_unfreeze:
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = True
                print('BERT embeddings unfrozen')

            # training
            train_metrics = self.train_evaluate_epoch(epoch, n_epoch, train_iter, 'train')
            # append history
            self.epoch_metrics['training']['epoch_{}'.format(epoch)] = train_metrics
            if valid_iter:
                # validation
                valid_metrics = self.train_evaluate_epoch(epoch, n_epoch, valid_iter, 'valid')
                # append_history
                self.epoch_metrics['validation']['epoch_{}'.format(epoch)] = valid_metrics
                # save best
                validation_f1 = np.mean(valid_metrics['f1_score'])
                if validation_f1 >= best_validation_f1:
                    best_validation_f1 = validation_f1
                    self.save_state_to_cache('best_validation_f1')

            if epoch >= bert_unfreeze:
                self.scheduler.step()
    
    
    def test(self, test_iter, test_path):
        ''' evaluates the test set '''
        metrics, tokens, attention, valid, label, prediction = self.train_evaluate_epoch(0, 1, test_iter, 'test')
        torch.save((metrics, tokens, attention, valid, label, prediction), test_path)
        return metrics, tokens, attention, valid, label, prediction
    

    def predict(self, predict_iter, predict_path):
        ''' evaluates the prediction set '''
        tokens, attention, valid, prediction, logits = self.train_evaluate_epoch(0, 1, predict_iter, 'predict')
        torch.save((tokens, attention, valid, prediction, logits), predict_path)
        return tokens, attention, valid, prediction, logits
