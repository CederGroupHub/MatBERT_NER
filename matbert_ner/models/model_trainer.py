import copy
import numpy as np
from tqdm import tqdm
import torch
from torch.optim.lr_scheduler import LambdaLR
from transformers import AdamW
from torchtools.optim import RangerLars, Ralamb, Ranger, Novograd, RAdam, Lamb, Lookahead
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import accuracy_score, classification_report
import json


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class StateCacher(object):
    '''
    An object that caches the state of a training model (parameters and optimizer)
    '''
    def __init__(self):
        '''
        Initializes the stach cacher object
            Arguments:
                None
            Returns:
                None
        '''
        # initialize empty dictionary of cached states
        self.cached = {}


    def store(self, key, state_dict):
        '''
        Caches a state
            Arguments:
                key: Identifier for state
                state_dict: Dictionary of states
            Returns:
                None
        '''
        # update dictionary of states
        self.cached.update({key: copy.deepcopy(state_dict)})


    def retrieve(self, key):
        '''
        Retrieves a cached state
            Arguments:
                key: Identifier for state
            Returns:
                State dictionary
        '''
        # raise error if key does not exist
        if key not in self.cached:
            raise KeyError('target {} was not cached.'.format(key))
        # return state dictionary
        return self.cached.get(key)


class NERTrainer(object):
    '''
    NER Trainer object for BERT NER
    '''
    def __init__(self, model, device):
        '''
        Initializes NER Trainer
            Arguments:
                model: Model to be trained
                device: Computation device
            Returns:
                NER Trainer object
        '''
        # dictionaries of ids and tokens for [CLS] and [SEP]
        self.cls_dict = {'id': 2, 'token': '[CLS]'}
        self.sep_dict = {'id': 3, 'token': '[SEP]'}
        # gradient clipping cutoff
        self.max_grad_norm = 1.0
        # computation device
        self.device = device
        # send model to device
        self.model = model.to(self.device)
        # initialize state cacher
        self.state_cacher = StateCacher()
        # initialize optimizer and scheduler
        self.optimizer = None
        self.scheduler = None
        # metric reporting mode
        self.metric_mode = 'strict'
        # set metric scheme according to labeling scheme used by model
        if self.model.scheme == 'IOB1':
            self.metric_scheme = IOB1
        elif self.model.scheme == 'IOB2':
            self.metric_scheme = IOB2
        elif self.model.scheme == 'IOBES':
            self.metric_scheme = IOBES
        # past training epochs
        self.past_epoch = 0
    

    def save_state(self, state_path, optimizer=True):
        '''
        Saves the state of the model and optimizer to file
            Arguments:
                state_path: Path to save the state to
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # state consists of classes and model parameter state dictionary
        state = {'classes': self.model.classes,
                 'model_state_dict': self.model.state_dict()}
        # if optimizer, include state dictionary
        if optimizer:
            state['optimizer_state_dict'] = self.optimizer.state_dict()
        # save to path
        torch.save(state, state_path)
    

    def load_state(self, state_path, optimizer=True):
        '''
        Loads the state of the model and optimizer from file
            Arguments:
                state_path: Path to load the state from
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # load checkpoint and map to device
        checkpoint = torch.load(state_path, map_location=torch.device(self.device))
        # set classes in model
        self.model.classes = checkpoint['classes']
        # rebuild model layers
        self.model.build_model()
        # send model to device
        self.model.to(self.device)
        # load model parameters from state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # if optimizer, load state
        if optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


    def save_state_to_cache(self, key, optimizer=True):
        '''
        Saves state to cache
            Arguments:
                key: Identification key
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # cache model state dictionary with key
        self.state_cacher.store('model_state_dict_{}'.format(key), self.model.state_dict())
        # if optimizer, cache optimizer state dictionary with key
        if optimizer:
            self.state_cacher.store('optimizer_state_dict_{}'.format(key), self.optimizer.state_dict())
    

    def load_state_from_cache(self, key, optimizer=True):
        '''
        Loads state from cache
            Arguments:
                key: Identification key
                optimizer: Boolean controlling whether to save the optimizer state
            Returns:
                None
        '''
        # load model parameters from state dictionary with key
        self.model.load_state_dict(self.state_cacher.retrieve('model_state_dict_{}'.format(key)))
        # if optimizer, load optimzier state dictionary with key
        if optimizer:
            self.optimizer.load_state_dict(self.state_cacher.retrieve('optimizer_state_dict_{}'.format(key)))
    

    def save_history(self, history_path):
        ''' 
        Save training histories to file
            Arguments:
                history_path: Path to save the history to
            Returns:
                None
        '''
        # save epoch metrics
        with open(history_path, 'w') as f:
            f.write(json.dumps(self.epoch_metrics, indent=2, cls=NpEncoder))
    

    def load_history(self, history_path):
        '''
        Load training histories from file
            Arguments:
                history_path: Path to load the history from
            Returns:
                None
        '''
        # load epoch metrics from path
        with open(history_path, 'r') as f:
            self.epoch_metrics = json.load(f)
        # set past epochs
        self.past_epoch = len(self.epoch_metrics['training'].keys())


    def return_history(self):
        '''
        Return history
            Arguments:
                None
            Returns:
                Dictionary of epoch metrics
        '''
        # return epoch metrics
        return self.epoch_metrics
    

    def init_optimizer(self, optimizer_name, elr, tlr, clr, weight_decay):
        '''
        Initialize optimizer
            Arguments:
                optimizer_name: Name of optimizer (adamw or rangerlars)
                elr: BERT embedding learning rate
                tlr: BERT encoder learning rate
                clr: Classifier learning rate (including classifier and CRF)
            Returns:
                None
        ''' 
        # optimizer dict
        optimizer_options = optimizer_name.split('_')
        optimizer_name = optimizer_options[0]
        if len(optimizer_options) > 1:
            if optimizer_options[1] == 'lookahead':
                la = True
            else:
                la = False
        else:
            la = False
        optimizers = {'adamw': AdamW, 'rangerlars': RangerLars,
                      'ralamb': Ralamb, 'ranger': Ranger,
                      'novograd': Novograd, 'radam': RAdam, 'lamb': Lamb}
        # default to AdamW if invalid optimizer name provided
        if optimizer_name not in optimizers.keys():
            optimizer_name = 'adamw'
            print('Reverted to default optimizer (AdamW)')
        # construct optimizer
        bert_embeddings_params = self.model.bert.embeddings.named_parameters()
        bert_encoder_params = self.model.bert.encoder.named_parameters()
        classifier_params = self.model.classifier.named_parameters()
        crf_params = self.model.crf.named_parameters()
        no_decay = ['bias', 'gamma', 'beta']
        optimizer=optimizers[optimizer_name]([{'params': [p for n, p in bert_embeddings_params if not any(nd in n for nd in no_decay)], 'lr': elr, 'weight_decay': weight_decay},
                                              {'params': [p for n, p in bert_embeddings_params if any(nd in n for nd in no_decay)], 'lr': elr, 'weight_decay': 0.0},
                                              {'params': [p for n, p in bert_encoder_params if not any(nd in n for nd in no_decay)], 'lr': tlr, 'weight_decay': weight_decay},
                                              {'params': [p for n, p in bert_encoder_params if any(nd in n for nd in no_decay)], 'lr': tlr, 'weight_decay': 0.0},
                                              {'params': [p for n, p in classifier_params if not any(nd in n for nd in no_decay)], 'lr': clr, 'weight_decay': weight_decay},
                                              {'params': [p for n, p in classifier_params if any(nd in n for nd in no_decay)], 'lr': clr, 'weight_decay': 0.0},
                                              {'params': [p for n, p in crf_params if not any(nd in n for nd in no_decay)], 'lr': clr, 'weight_decay': weight_decay},
                                              {'params': [p for n, p in crf_params if any(nd in n for nd in no_decay)], 'lr': clr, 'weight_decay': 0.0}])
        if la:
            self.optimizer = Lookahead(base_optimizer=optimizer, k=10, alpha=0.5)
        else:
            self.optimizer = optimizer
    

    def init_scheduler(self, n_epoch, bert_unfreeze, function_name='exponential'):
        '''
        Initializes learning rate scheduler
            Arguments:
                n_epoch: Number of training epochs
                bert_unfreeze: Epoch at which any BERT layers are unfrozen
                function_name: Name of scheduling funtion (linear, exponential, or cosine)
            Returns:
                None
        '''
        # dictionary of functions
        functions = {'linear': lambda epoch: (n_epoch-bert_unfreeze-epoch)/(n_epoch-bert_unfreeze),
                     'exponential': lambda epoch: 0.1**(epoch/(n_epoch-bert_unfreeze-1)),
                     'cosine': lambda epoch: 0.5*(1+np.cos(epoch/(n_epoch-bert_unfreeze)*np.pi))}
        # default to linear if invalid function name provided
        if function_name not in functions.keys():
            function_name = 'linear'
            print('Reverted to default scheduling function (linear)')
        # construct scheduler
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=functions[function_name], verbose=True)
    

    def construct_valid_inputs(self, inputs):
        '''
        Construct valid input ids, valid label ids, and valid attention masks
            Arguments:
                inputs: BERTNER inputs
            Returns:
                Valid input ids, valid label ids, and valid attention masks as lists of NumPy arrays
        '''
        # size of arrays
        batch_size, max_len = inputs['valid_mask'].shape
        # initialize empty arrays
        valid = {key: np.zeros((batch_size, max_len), dtype=int) for key in ['input_ids', 'label_ids', 'attention_mask']}
        # for batch index
        for i in range(batch_size):
            # set valid index to 0
            k = 0
            # for sequence index
            for j in range(max_len):
                # if sequence index is valid
                if inputs['valid_mask'][i][j].item() == 1:
                    # assign valid values at (batch index, valid index) to input values at (batch index, sequence index)
                    for key in valid.keys():
                        valid[key][i, k] = inputs[key][i][j].item()
                    # increment valid index
                    k += 1
        # return valid inputs
        return tuple([list(valid[key]) for key in valid.keys()])
    

    def process_labels(self, inputs, prediction_ids):
        '''
        Process labels for metric evaluation accounting for invalid indices while filtering out [CLS] and [SEP] tokens
            Arguments:
                inputs: BERTNER inputs
                prediction_ids: BERTNER output predictions
            Returns:
                Dictionary of valid labels and predictions
        '''
        # construct valid inputs
        valid_input_ids, valid_label_ids, valid_attention_mask = self.construct_valid_inputs(inputs)
        # for sequence in batch
        for i in range(len(valid_attention_mask)):
            # identify padding indices
            idx = np.where(valid_attention_mask[i] == 0)
            # trim padding indices
            valid_input_ids[i] = np.delete(valid_input_ids[i], idx, axis=0)
            valid_label_ids[i] = np.delete(valid_label_ids[i], idx, axis=0)
        # convert labels and predictions into classes given indices as long as the corresponding input ids do not correspond to [CLS] or [SEP]
        labels = [[self.model.classes[vli] for vli, vii in zip(vlis, viis) if vii not in (self.cls_dict['id'], self.sep_dict['id'])] for vlis, viis in zip(valid_label_ids, valid_input_ids)]
        predictions = [[self.model.classes[pi] for pi, vii in zip(pis, viis) if vii not in (self.cls_dict['id'], self.sep_dict['id'])] for pis, viis in zip(prediction_ids, valid_input_ids)]
        # return dictionary of valid labels and predictions
        return {'labels': labels, 'predictions': predictions}
    

    def process_ids(self, ids, input_ids, attention_mask, valid_mask, prediction_ids):
        '''
        Processes ids into tokens and labels
            Arguments:
                input_ids: Sequence token ids
                attention_mask: Sequence attention masks
                valid_mask: Sequence valid masks
                prediction_ids: Sequence prediction ids
            Returns:
                Dictionary of text and annotations by word, sentence, paragraph e.g. [[[{'text': text, 'annotation': annotation},...],...],...]
        '''
        # for entry index
        for i in range(len(attention_mask)):
            # identify padding indices
            idx = np.where(np.array(attention_mask[i]) == 0)[0]
            # trim padding indices
            input_ids[i] = np.delete(np.array(input_ids[i]), idx, axis=0)
            valid_mask[i] = np.delete(np.array(valid_mask[i]), idx, axis=0)
        # convert token ids to tokens for each entry
        toks = [self.model.tokenizer.convert_ids_to_tokens(sequence) for sequence in input_ids]
        # convert label ids to classifications for each entry
        lbls = [[self.model.classes[l] for l in sequence] for sequence in prediction_ids]
        # initialize empty lists of entries for tokens and labels
        # ctoks for c(ombined)tok(en)s, referencing that we will be merging the tokens split by the BERT tokenizer
        # slbls for s(ingle)l(a)b(e)ls, referencing that we often only have a single label for multiple subtokens
        pids = []
        ctoks = []
        slbls = []
        # for index in entries
        for i in range(len(valid_mask)):
            # initialize a list of an empty list, each sublist will be a sentence
            ctok = [[]]
            slbl = [[]]
            # initialize special subindices
            # sentence index (only stepped forward on new sentence)
            k = 0
            # index within sentence (only stepped forward within sentence and reset on new sentence)
            u = 0
            # index in labels (only stepped forward on valid indices)
            v = 0
            # for index in sequence
            for j in range(len(valid_mask[i])):
                # if the index is valid
                if valid_mask[i][j] == 1:
                    # if the index is a cls token, pass
                    if toks[i][j] == self.cls_dict['token']:
                        pass
                    # if the index is a sep token
                    elif toks[i][j] == self.sep_dict['token']:
                        # if the current index is not the last index in the sequence
                        if j < len(valid_mask[i])-1:
                            # append new empty sentence
                            ctok.append([])
                            slbl.append([])
                            # step sentence index forward
                            k += 1
                            # reset index within sentence to zero
                            u = 0
                    # if valid token and not special token
                    else:
                        # append token to combined tokens
                        ctok[k].append(toks[i][j])
                        # append label to single labels
                        slbl[k].append(lbls[i][v] if lbls[i][v] == 'O' else lbls[i][v].split('-')[1])
                        # step index within sentence forward
                        u += 1
                    # step index within labels
                    v += 1
                # if not valid index
                else:
                    # if token is suffix (as determined by BERT tokenizer with ##)
                    if '##' in toks[i][j]:
                        # combine with prior token while filtering out the suffix indication (##)
                        ctok[k][u-1] += toks[i][j].replace('##', '')
                    # otherwise
                    else:
                        # combine with prior token
                        ctok[k][u-1] += toks[i][j]
            # append sentences
            pids.append(ids[i])
            ctoks.append(ctok)
            slbls.append(slbl)
        # construct dictionary of (text, annotation) pairs under tokens and a dictionary of entities under entities
        annotations = [{'id': id, 'tokens': [[{'text': t, 'annotation': l} for t, l in zip(ctoks_sequence, slbls_sequence)]
                                   for ctoks_sequence, slbls_sequence in zip(ctoks_entry, slbls_entry)]}
                       for id, ctoks_entry, slbls_entry in zip(pids, ctoks, slbls)]
        return annotations
    

    def process_summaries(self, annotations):
        # class types
        class_types = sorted(list(set([class_.split('-')[1] for class_ in self.model.classes if class_ != 'O'])))
        # for entry
        for annotation in annotations:
            # initialize empty dictionary of extracted entities
            entry_entities = {class_type: set([]) for class_type in class_types}
            # for sentence in entry
            for sentence in annotation['tokens']:
                # extract beginning indices for entities
                entity_idx = [k for k in range(len(sentence)) if sentence[k]['annotation'] != sentence[k-1]['annotation']]
                # append ending sequence
                entity_idx.append(len(sentence))
                # for entity beginning index
                for k in range(len(entity_idx)):
                    # if not last index (last index in sequence)
                    if k != len(entity_idx)-1:
                        # if entity is not outside
                        if sentence[entity_idx[k]]['annotation'] != 'O':
                            # append joined entity to dictionary
                            entry_entities[sentence[entity_idx[k]]['annotation']].add(' '.join([sentence[u]['text'] for u in range(entity_idx[k], entity_idx[k+1])]))
            # append entry entity dictionary
            annotation['entities'] = {class_type: sorted(list(set(entry_entities[class_type]))) for class_type in class_types}
        return annotations
    

    def iterate_batches(self, epoch, n_epoch, iterator, mode):
        '''
        Iterates through batches in epoch
            Arguments:
                epoch: Current epoch
                n_epoch: Total number of epochs
                iterator: Dataloader
                mode: Model mode e.g. train, evaluate, test, or predict
            Returns:
                mode == train or validate: metrics
                mode == test: metrics and test results
                mode == predict: prediction results
        '''
        # if mode is not prediction, initialize empty list for metrics
        if mode != 'predict':
            metrics = []
        # if mode is test, initialize dictionary of labels and predictions
        if mode == 'test':
            test_results = {'labels': [], 'predictions': []}
        # if mode is predict, initialize dictionary of input_ids, attention_masks, valid_masks, and prediction_ids
        if mode == 'predict':
            prediction_results = {'ids': [], 'pts': [], 'input_ids': [], 'attention_mask': [], 'valid_mask': [], 'prediction_ids': []}
        # initialize batch range
        batch_range = tqdm(iterator, desc='')
        # for batch
        for batch in batch_range:
            # collect inputs from batch
            ids = batch[0].cpu().numpy()
            pts = batch[1].cpu().numpy()
            inputs = {'input_ids': batch[2].to(self.device, non_blocking=True),
                      'attention_mask': batch[4].to(self.device, non_blocking=True),
                      'valid_mask': batch[5].to(self.device, non_blocking=True),
                      'device': self.device}
            # collect labels if the mode is not predict
            if mode != 'predict':
                inputs['label_ids'] = batch[3].to(self.device, non_blocking=True)

            # zero out prior gradients for training
            if mode == 'train':
                self.optimizer.zero_grad()

            # if mode is not predict, collect loss and prediction ids and then process labels
            if mode != 'predict':
                loss, prediction_ids = self.model.forward(**inputs)
                batch_results = self.process_labels(inputs, prediction_ids)
            # if mode is predict, only collect prediction ids
            else:
                prediction_ids = self.model.forward(**inputs)                
            
            # if mode is test, extend the list of batch results in the test results dictionary for the labels and predictions keys
            if mode == 'test':
                for key in test_results.keys():
                    test_results[key].extend(batch_results[key])
            # if mode is predict, extend lists of input_ids, attention_masks, valid_masks, and prediction_ids by keys
            if mode == 'predict':
                for key in prediction_results.keys():
                    if key == 'ids':
                        prediction_results[key].extend(ids)
                    elif key == 'pts':
                        prediction_results[key].extend(pts)
                    elif key == 'prediction_ids':
                        prediction_results[key].extend(prediction_ids)
                    else:
                        prediction_results[key].extend(list(inputs[key].cpu().numpy()))

            # if mode is not predice
            if mode != 'predict':
                # generate classification report from labels and predictions (with filtered out special tokens)
                report = classification_report(batch_results['labels'], batch_results['predictions'], mode=self.metric_mode, scheme=self.metric_scheme, output_dict=True)
                # add accuracy score to report
                report['accuracy'] = accuracy_score(batch_results['labels'], batch_results['predictions'])
                # add loss to report
                report['loss'] = loss.item()
                # append report to metrics
                metrics.append(report)
                # calculate rolling means for precision, recall and f1-score
                means = {m: np.mean([r['micro avg'][m] for r in metrics]) for m in ['precision', 'recall', 'f1-score']}
                # calculate rolling means for accuracy and loss
                means['accuracy'] = np.mean([r['accuracy'] for r in metrics])
                means['loss'] = np.mean([r['loss'] for r in metrics])

            # backpropagate the gradients and step the optimizer forward
            if mode == 'train':
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()

            # if mode is not predict
            if mode != 'predict':
                del loss
                # display epoch progress, mode, and rolling averages alongside batch progress
                msg = '| epoch: {:d}/{:d} | {} | loss: {:.4f} | accuracy: {:.4f} | precision: {:.4f} | recall: {:.4f} | f1-score: {:.4f} |'
                info = (self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode, means['loss'], means['accuracy'], means['precision'], means['recall'], means['f1-score'])
            # if mode is predict
            else:
                # display epoch progress, mode, and batch progress
                msg = '| epoch: {:d}/{:d} | {} |'
                info = (self.past_epoch+epoch+1, self.past_epoch+n_epoch, mode)
            # set progress bar description
            batch_range.set_description(msg.format(*info))
        # return statements
        if mode == 'test':
            return metrics, test_results
        elif mode != 'predict':
            return metrics
        else:
            return prediction_results
    

    def train_evaluate_epoch(self, epoch, n_epoch, iterator, mode):
        '''
        Trains or evaluate the model for an epoch
            Arguments:
                epoch: Current epoch
                n_epoch: Total number of epochs
                iterator: Dataloader
                mode: Model mode e.g. train, evaluate, test, or predict
            Returns:
                mode == train or validate: metrics
                mode == test: metrics and test results
                mode == predict: prediction results
        '''
        # if training
        if mode == 'train':
            # make sure the model is set to train
            self.model.train()
            # train all of the batches and collect the metrics
            metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # if validating
        elif mode == 'valid':
            # make sure the model is set to evaluate
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the metrics
                metrics = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # if testing
        elif mode == 'test':
            # make sure the model is set to evaluate
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the metrics
                metrics, test_results = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # if predicting
        elif mode == 'predict':
            # make sure the model is set to evaluate
            self.model.eval()
            # turn off gradients for evaluating
            with torch.no_grad():
                # evaluate all of the batches and collect the prediction results
                prediction_results = self.iterate_batches(epoch, n_epoch, iterator, mode)
        # return statements
        if mode == 'test':
            return metrics, test_results
        elif mode != 'predict':
            return metrics
        else:
            return prediction_results
    

    def train(self, n_epoch, train_iter, valid_iter, embedding_unfreeze, encoder_schedule, scheduling_function, save_dir=None, use_cache=False):
        '''
        Trains the model with validation if a validation iterator is provided
            Arguments:
                n_epoch: Total number of epochs
                train_iter: Training dataloader
                valid_iter: Validation dataloader
                embedding_unfreeze: Epoch when the BERT embeddings are unfrozen
                encoder_schedule: List of number of BERT encoders to unfreeze per epoch
                scheduling_function: Learning rate schedule function
                save_dir: Save directory for model and optimizer state
                use_cache: Boolean that controls whether to use the cache for saving the model/optimizer state. If False, states are saved to disk at save_dir
            Returns:
                None
        '''
        # initialize dictionary of epoch metrics
        self.epoch_metrics = {'training': {}}
        if valid_iter is not None:
            self.epoch_metrics['validation'] = {}
        
        # first epoch with at least one unfrozen BERT encoder
        encoder_unfreeze = next((i for i, n in enumerate(encoder_schedule) if n), n_epoch)
        # first epoch with at least one unfrozen BERT layer (not counting pooler)
        bert_unfreeze = encoder_unfreeze if encoder_unfreeze < embedding_unfreeze else embedding_unfreeze
        # initialize scheduler
        self.init_scheduler(n_epoch, bert_unfreeze, scheduling_function)

        # last encoder index
        last_encoder_layer = 11
        # empty encoder schedule dictionary
        expanded_encoder_schedule = {}
        # for epoch
        for epoch in range(n_epoch):
            # initialize empty list of encoder indices for epoch
            expanded_encoder_schedule['epoch_{}'.format(epoch)] = []
            # for number of encoders to be unfrozen during epoch
            for layer in range(encoder_schedule[epoch]):
                # append encoder index
                expanded_encoder_schedule['epoch_{}'.format(epoch)].append(last_encoder_layer)
                # decrement last encoder index
                last_encoder_layer -= 1

        # freeze BERT embeddings and encoders
        for param in self.model.bert.embeddings.parameters():
            param.requires_grad = False
        for param in self.model.bert.encoder.parameters():
            param.requires_grad = False
        print('BERT embeddings and encoders frozen')
        print('CRF, and Classifier unfrozen')
        # initialize best validation f1
        best_validation_f1 = 0.0

        # for each epoch
        for epoch in range(n_epoch):
            # for each BERT encoder to be unfrozen this epoch
            for layer_index in expanded_encoder_schedule['epoch_{}'.format(epoch)]:
                # unfreeze BERT encoder parameters
                for param in self.model.bert.encoder.layer[layer_index].parameters():
                    param.requires_grad = True
                print('BERT encoder {} unfrozen'.format(layer_index))
            # unfreeze BERT embedding at appropriate epoch
            if epoch == embedding_unfreeze:
                for param in self.model.bert.embeddings.parameters():
                    param.requires_grad = True
                print('BERT embeddings unfrozen')

            # training
            train_metrics = self.train_evaluate_epoch(epoch, n_epoch, train_iter, 'train')
            # append history
            self.epoch_metrics['training']['epoch_{}'.format(self.past_epoch+epoch)] = train_metrics
            if valid_iter:
                # validation
                valid_metrics = self.train_evaluate_epoch(epoch, n_epoch, valid_iter, 'valid')
                # append_history
                self.epoch_metrics['validation']['epoch_{}'.format(self.past_epoch+epoch)] = valid_metrics
                # save best
                validation_f1 = np.mean([batch_metrics['micro avg']['f1-score'] for batch_metrics in valid_metrics])
                if validation_f1 >= best_validation_f1:
                    best_validation_f1 = validation_f1
                    if use_cache:
                        self.save_state_to_cache('best')
                    else:
                        self.save_state(save_dir+'best.pt')

            # if BERT is unfrozen and the epoch is not the last, step the scheduler forward
            if epoch >= bert_unfreeze and epoch < n_epoch-1:
                self.scheduler.step()
    
    
    def test(self, test_iter, test_path=None, state_path=None):
        '''
        Evaluates the tests set and saves the predictions alongside the ground truths
            Arguments:
                test_iter: Test dataloader
                test_path: Path to save the test results to
                state_path: Path to load the model state from
            Returns:
                metrics and test results
        '''
        # if state path provided, load state (excluding optimizer)
        if state_path is not None:
            self.load_state(state_path, optimizer=False)
        # evaluate the test set
        metrics, test_results = self.train_evaluate_epoch(0, 1, test_iter, 'test')
        # save the test metrics and results
        if test_path is not None:
            with open(test_path, 'w') as f:
                f.write(json.dumps({'metrics': metrics, 'results': test_results}, indent=2, cls=NpEncoder))
        # return the test metrics and results
        return metrics, test_results
    

    def merge_split_entries(self, prediction_results):
        unique_ids = []
        for id in prediction_results['ids']:
            if id not in unique_ids:
                unique_ids.append(id)
        merged_prediction_results = {key: [] for key in prediction_results.keys() if key != 'pts'}
        for id in unique_ids:
            merged_prediction_results['ids'].append(id)
            id_pts_ind = np.where(prediction_results['ids'] == id)[0]
            id_pts_ind = id_pts_ind[np.argsort(np.array(prediction_results['pts'])[id_pts_ind])]
            for i, u in enumerate(id_pts_ind):
                if i == 0:
                    for key in prediction_results.keys():
                        if key not in ['ids', 'pts']:
                            merged_prediction_results[key].append(list(prediction_results[key][u]))
                else:
                    for key in prediction_results.keys():
                        if key not in ['ids', 'pts']:
                            merged_prediction_results[key][-1].extend(list(prediction_results[key][u]))
        return merged_prediction_results
    

    def predict(self, predict_iter, original_data=None, state_path=None, predict_path=None, return_full_dict=False):
        '''
        Predicts classifications for a dataset
            Arguments:
                predict_iter: Prediction dataloader
                original_data: Original data before pre-processing
                state_path: Path to load the model state from
                predict_path: Path to save the predictions to
                return_full_dict: Toggle for returning full JSON entries or only the detected entities
            Returns:
                Dictionary of text and annotations by word, sentence, paragraph e.g. [[[{'text': text, 'annotation': annotation},...],...],...]
                  or dictionary of entity summaries
        '''
        # if state path provided, load state (excluding optimizer)
        if state_path is not None:
            self.load_state(state_path, optimizer=False)
        # evaluate the prediction set
        prediction_results = self.train_evaluate_epoch(0, 1, predict_iter, 'predict')
        # process the predictions into annotations
        prediction_results = self.merge_split_entries(prediction_results)
        annotations = self.process_ids(prediction_results['ids'], prediction_results['input_ids'], prediction_results['attention_mask'],
                                       prediction_results['valid_mask'], prediction_results['prediction_ids'])
        if original_data is not None:
            output_annotations = []
            annotation_dict = {}
            for annotation in annotations:
                annotation_dict[annotation['id']] = {'tokens': annotation['tokens']}
            for original in original_data:
                annotation = annotation_dict[original['id']]
                output_annotation = {'id': original['id'], 'meta': original['meta'], 'tokens': annotation['tokens']}
                for original_sentence, output_annotation_sentence  in zip(original['tokens'], output_annotation['tokens']):
                    for original_token, output_annotation_token in zip(original_sentence['text'], output_annotation_sentence):
                        output_annotation_token['text'] = original_token
                output_annotations.append(output_annotation)
        else:
            output_annotations = annotations
        annotations = self.process_summaries(output_annotations)
        # save annotations
        if predict_path is not None:
            with open(predict_path, 'w') as f:
                f.write(json.dumps(annotations, indent=2))
        # return the annotations
        if return_full_dict:
            return annotations
        else:
            return [annotation['entities'] for annotation in annotations]
