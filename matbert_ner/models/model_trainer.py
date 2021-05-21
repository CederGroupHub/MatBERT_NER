import copy
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW
from torchtools.optim import RangerLars
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import accuracy_score, classification_report


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
    def __init__(self, model, device):
        '''
        class for basic functions common to the trainer objects used in this project

        model: the model to be trained
        device: torch device
        '''
        self.cls_dict = {'id': 2, 'token': '[CLS]'}
        self.sep_dict = {'id': 3, 'token': '[SEP]'}
        self.max_grad_norm = 1.0
        self.device = device
        # send model to device
        self.model = model.to(self.device)
        self.state_cacher = StateCacher()
        self.optimizer = None
        self.scheduler = None
        self.metric_mode = 'strict'
        if self.model.scheme == 'IOB1':
            self.metric_scheme = IOB1
        elif self.model.scheme == 'IOB2':
            self.metric_scheme = IOB2
        elif self.model.scheme == 'IOBES':
            self.metric_scheme = IOBES
        self.past_epoch = 0
    

    def save_state(self, state_path, optimizer=True):
        ''' saves entire model state to file '''
        state = {'classes': self.model.classes,
                 'model_state_dict': self.model.state_dict()}
        if optimizer:
            state['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(state, state_path)
    

    def load_state(self, state_path, optimizer=True):
        ''' loads entire model state from file '''
        checkpoint = torch.load(state_path)
        self.model.classes = checkpoint['classes']
        self.model.build_model()
        self.model.to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def save_state_to_cache(self, key, optimizer=True):
        self.state_cacher.store('model_state_dict_{}'.format(key), self.model.state_dict())
        if optimizer:
            self.state_cacher.store('optimizer_state_dict_{}'.format(key), self.optimizer.state_dict())
    

    def load_state_from_cache(self, key, optimizer=True):
        self.model.load_state_dict(self.state_cacher.retrieve('model_state_dict_{}'.format(key)))
        if optimizer:
            self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer_state_dict_{}'.format(key)))
    

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
    

    def init_optimizer(self, optimizer_name, elr, tlr, clr):
        self.optimizer_name = optimizer_name
        self.elr, self.tlr, self.clr = elr, tlr, clr
        optimizers = {'adamw': AdamW, 'rangerlars': RangerLars}
        self.optimizer = optimizers[optimizer_name]([{'params': self.model.bert.embeddings.parameters(), 'lr': self.elr},
                                                     {'params': self.model.bert.encoder.parameters(), 'lr': self.tlr},
                                                     {'params': self.model.bert.pooler.parameters(), 'lr': self.clr},
                                                     {'params': self.model.classifier.parameters(), 'lr': self.clr},
                                                     {'params': self.model.crf.parameters(), 'lr': self.clr}])
    

    def init_scheduler(self, n_epoch, bert_unfreeze, function='linear'):
        functions = {'linear': lambda epoch: (n_epoch-bert_unfreeze-epoch)/(n_epoch-bert_unfreeze),
                     'exponential': lambda epoch: 0.1**(epoch/(n_epoch-1)),
                     'cosine': lambda epoch: 0.5*(1+np.cos(epoch/n_epoch*np.pi))}
        if function not in functions.keys():
            function = 'linear'
            print('Reverted to default scheduling function (linear)')
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=functions[function], verbose=True)
    

    def construct_valid_inputs(self, inputs):
        batch_size, max_len = inputs['valid_mask'].shape
        valid_input_ids = np.zeros((batch_size, max_len), dtype=int)
        valid_label_ids = np.zeros((batch_size, max_len), dtype=int)
        valid_attention_mask = np.zeros((batch_size, max_len), dtype=int)
        for i in range(batch_size):
            jj = 0
            for j in range(max_len):
                if inputs['valid_mask'][i][j].item() == 1:
                    valid_input_ids[i, jj] = inputs['input_ids'][i][j].item()
                    valid_label_ids[i, jj] = inputs['label_ids'][i][j].item()
                    valid_attention_mask[i, jj] = inputs['attention_mask'][i][j].item()
                    jj += 1
        return list(valid_input_ids), list(valid_label_ids), list(valid_attention_mask)
    

    def process_labels(self, inputs, prediction_ids):
        valid_input_ids, valid_label_ids, valid_attention_mask = self.construct_valid_inputs(inputs)
        for i in range(len(valid_attention_mask)):
            idx = np.where(valid_attention_mask[i] == 0)
            valid_input_ids[i] = np.delete(valid_input_ids[i], idx, axis=0)
            valid_label_ids[i] = np.delete(valid_label_ids[i], idx, axis=0)
        labels = [[self.model.classes[vli] for vli, vii in zip(vlis, viis) if vii not in (self.cls_dict['id'], self.sep_dict['id'])] for vlis, viis in zip(valid_label_ids, valid_input_ids)]
        predictions = [[self.model.classes[pi] for pi, vii in zip(pis, viis) if vii not in (self.cls_dict['id'], self.sep_dict['id'])] for pis, viis in zip(prediction_ids, valid_input_ids)]
        results = {'labels': labels, 'predictions': predictions}
        return results
    

    def process_ids(self, input_ids, attention_mask, valid_mask, prediction_ids):
        for i in range(len(attention_mask)):
            idx = np.where(attention_mask[i] == 0)
            input_ids[i] = np.delete(input_ids[i], idx, axis=0)
            valid_mask[i] = np.delete(valid_mask[i], idx, axis=0)
        toks = [self.model.tokenizer.convert_ids_to_tokens(sequence) for sequence in input_ids]
        lbls = [[self.model.classes[l] for l in sequence] for sequence in prediction_ids]
        ctoks = []
        slbls = []
        for i in range(len(valid_mask)):
            ctok = [[]]
            slbl = [[]]
            ii = 0
            jj = 0
            kk = 0
            for j in range(len(valid_mask[i])):
                if valid_mask[i][j] == 1:
                    if toks[i][j] == self.cls_dict['token']:
                        pass
                    elif toks[i][j] == self.sep_dict['token']:
                        if j < len(valid_mask[i])-1:
                            ctok.append([])
                            slbl.append([])
                            ii += 1
                            jj = 0
                    else:
                        ctok[ii].append(toks[i][j])
                        if lbls[i][kk] == 'O':
                            slbl[ii].append(lbls[i][kk])
                        else:
                            slbl[ii].append(lbls[i][kk])
                        jj += 1
                    kk += 1
                else:
                    if '##' in toks[i][j]:
                        ctok[ii][jj-1] += toks[i][j].replace('##', '')
                    else:
                        ctok[ii][jj-1] += toks[i][j]
            ctoks.append(ctok)
            slbls.append(slbl)
        annotations = [[[{'text': t, 'annotation': l} for t, l in zip(ctoksw, slblsw)] for ctoksw, slblsw in zip(ctoksp, slblsp)] for ctoksp, slblsp in zip(ctoks, slbls)]
        return annotations
    

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
            metrics = []
        if mode == 'test':
            test_results = {'labels': [], 'predictions': []}
        if mode == 'predict':
            prediction_results = {'input_ids': [], 'attention_mask': [], 'valid_mask': [], 'prediction_ids': []}
        # initialize batch range
        batch_range = tqdm(iterator, desc='')
        for batch in batch_range:
            # fetch texts, characters, and tags from batch
            inputs = {'input_ids': batch[0].to(self.device, non_blocking=True),
                      'attention_mask': batch[2].to(self.device, non_blocking=True),
                      'valid_mask': batch[3].to(self.device, non_blocking=True),
                      'device': self.device}
            if mode != 'predict':
                inputs['label_ids'] = batch[1].to(self.device, non_blocking=True)

            # zero out prior gradients for training
            if mode == 'train':
                self.optimizer.zero_grad()

            # output depends on whether conditional random field is used for prediction/loss
            if mode != 'predict':
                loss, prediction_ids = self.model.forward(**inputs)
                batch_results = self.process_labels(inputs, prediction_ids)
            else:
                prediction_ids = self.model.forward(**inputs)
            
            if mode == 'test':
                for key in test_results.keys():
                    test_results[key].extend(batch_results[key])
            if mode == 'predict':
                for key in prediction_results.keys():
                    if key == 'prediction_ids':
                        prediction_results[key].extend(prediction_ids)
                    else:
                        prediction_results[key].extend(list(inputs[key].cpu().numpy()))

            if mode != 'predict':
                report = classification_report(batch_results['labels'], batch_results['predictions'], mode=self.metric_mode, scheme=self.metric_scheme, output_dict=True)
                report['accuracy'] = accuracy_score(batch_results['labels'], batch_results['predictions'])
                report['loss'] = loss.item()
                metrics.append(report)
                means = {m: np.mean([r['micro avg'][m] for r in metrics]) for m in ['precision', 'recall', 'f1-score']}
                means['accuracy'] = np.mean([r['accuracy'] for r in metrics])
                means['loss'] = np.mean([r['loss'] for r in metrics])

            # backpropagate the gradients and step the optimizer forward
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            if mode != 'predict':
                # display progress
                msg = '| epoch: {:d}/{:d} | {} | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1-score: {:.4f} |'
                info = (self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode, means['loss'], means['accuracy'], means['precision'], means['recall'], means['f1-score'])
            else:
                msg = ''
                info = ()
            batch_range.set_description(msg.format(*info))
        # return the batch losses and metrics
        if mode == 'test':
            return metrics, test_results
        elif mode != 'predict':
            return metrics
        else:
            return prediction_results
    

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
                metrics, test_results = self.iterate_batches(epoch, n_epoch, iterator, mode)
        elif mode == 'predict':
            # make sure the model is set to evaluate if it is evaluating
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the batch/epoch loss/metrics
                prediction_results = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # return batch/epoch loss/metrics
        if mode == 'test':
            return metrics, test_results
        elif mode != 'predict':
            return metrics
        else:
            return prediction_results
    

    def train(self, n_epoch, train_iter, valid_iter, embedding_unfreeze, encoder_schedule, scheduling_function, save_dir, use_cache):
        '''
        trains the model (with validation)

        n_epoch: number of training epochs
        '''
        self.epoch_metrics = {'training': {}}
        if valid_iter is not None:
            self.epoch_metrics['validation'] = {}

        encoder_unfreeze = next((i for i, n in enumerate(encoder_schedule) if n), n_epoch)
        bert_unfreeze = encoder_unfreeze if encoder_unfreeze < embedding_unfreeze else embedding_unfreeze
        self.init_scheduler(n_epoch, bert_unfreeze, scheduling_function)

        last_encoder_layer = 11
        expanded_encoder_schedule = {}
        for epoch in range(n_epoch):
            expanded_encoder_schedule['epoch_{}'.format(epoch)] = []
            for layer in range(encoder_schedule[epoch]):
                expanded_encoder_schedule['epoch_{}'.format(epoch)].append(last_encoder_layer)
                last_encoder_layer -= 1

        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.model.bert.encoder.parameters():
            param.requires_grad = False
        print('BERT embeddings and encoders frozen')
        print('CRF, Classifier, and BERT pooler unfrozen')

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
                validation_f1 = np.mean([batch_metrics['micro avg']['f1-score'] for batch_metrics in valid_metrics])
                if validation_f1 >= best_validation_f1:
                    best_validation_f1 = validation_f1
                    if use_cache:
                        self.save_state_to_cache('best')
                    else:
                        self.save_state(save_dir+'best.pt')

            if epoch >= bert_unfreeze and epoch < n_epoch-1:
                self.scheduler.step()
    
    
    def test(self, test_iter, test_path, state_path=None):
        ''' evaluates the test set '''
        if state_path is not None:
            self.load_state(state_path, optimizer=False)
        metrics, test_results = self.train_evaluate_epoch(0, 1, test_iter, 'test')
        torch.save((metrics, test_results), test_path)
        return metrics, test_results
    

    def predict(self, predict_iter, predict_path, state_path=None):
        ''' evaluates the prediction set '''
        if state_path is not None:
            self.load_state(state_path, optimizer=False)
        prediction_results = self.train_evaluate_epoch(0, 1, predict_iter, 'predict')
        annotations = self.process_ids(prediction_results['input_ids'], prediction_results['attention_mask'],
                                       prediction_results['valid_mask'], prediction_results['prediction_ids'])
        torch.save(annotations, predict_path)
        return annotations
