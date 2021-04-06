from models.bert_model import BertCRFNERModel
from models.bilstm_model import BiLSTMNERModel
from utils.data import NERData
import os
import glob
import json
import torch
import numpy as np

seeds = [2**x for x in np.arange(8)]
torch.backends.cudnn.deterministic = True

datafiles = {'solid_state': 'data/ner_annotations.json',
             'doping': 'data/doping.json',
             'impurityphase': 'data/impurityphase_fullparas.json',
             'aupnmorph': 'data/aunpmorph_annotations_fullparas.json'}

splits = np.arange(10, 85, 5)
n_epochs = 16
lr = 2e-4

device = 'cuda'
models = {'bilstm': 'bert-base-uncased',
          'bert': 'bert-base-uncased',
          'scibert': 'allenai/scibert_scivocab_uncased',
          'matbert': '/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased'}

model_names = ['bert']
data = ['solid_state', 'aunpmorph', 'doping']

for model_name in model_names:
    for data in data_names:
        for seed in seeds:
            torch.manual_seed(seed)
            configs = {'_{}_full_crf_iobes_{}_{}'.format(data, seed, split): {'full_finetuning': True, 'format': 'IOBES', 'split': [split/100, split/800, 0.1]} for split in splits}
            for alias, config in configs.items():
                save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

                ner_data = NERData(models[model_name], tag_format=config['format'])
                ner_data.preprocess(datafiles[data])

                train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(batch_size=32, train_frac=config['split'][0], val_frac=config['split'][1], dev_frac=config['split'][2], seed=seed)
                classes = ner_data.classes
                torch.save(save_dir+'classes.pt', classes)

                if model_name == 'bilstm':
                    ner_model = BiLSTMNERModel(modelname=models[model_name], classes=classes, tag_format=config['format'], device=device, lr=lr)
                else:
                    ner_model = BertCRFNERModel(modelname=models[model_name], classes=classes, tag_format=config['format'], device=device, lr=lr)
                print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
                print(ner_model.model)
                ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, dev_dataloader=dev_dataloader, save_dir=save_dir, full_finetuning=config['full_finetuning'])

                fs = glob.glob(save_dir+'epoch_*pt')
                for f in fs:
                    try:
                        os.remove(f)
                    except:
                        print('error while deleting file: {}'.format(f))
                try:
                    os.remove(save_dir+'best.pt')
                except:
                    print('error while deleting file: {}best.pt'.format(savedir))
