from models.bert_model import BertCRFNERModel
from models.bilstm_model import BiLSTMNERModel
from utils.data import NERData
import os
import glob
import json
import torch

seed = 256
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

# datafile = 'data/impurityphase_fullparas.json'
# datafile = 'data/aunpmorph_annotations_fullparas.json'
datafile = 'data/ner_annotations.json'
# datafile = 'data/doping.json'

split = (0.8, 0.1, 0.1)
n_epochs = 16
lr = 2e-4

device = "cuda"
models = {'bert': 'bert-base-uncased',
          'scibert': 'allenai/scibert_scivocab_uncased',
          'matbert': '/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased'}

configs = {}
configs['_ner_annotations_full_crf_iobes_{}'.format(seed)] = {'full_finetuning': True,
                                                              'format': 'IOBES'}
configs['_ner_annotations_full_crf_iob2_{}'.format(seed)] = {'full_finetuning': True,
                                                             'format': 'IOB2'}

configs['_ner_annotations_shallow_crf_iobes_{}'.format(seed)] = {'full_finetuning': False,
                                                                 'format': 'IOBES'}
configs['_ner_annotations_shallow_crf_iob2_{}'.format(seed)] = {'full_finetuning': False,
                                                                'format': 'IOB2'}

for alias, config in configs.items():
    for model_name in ['matbert', 'scibert', 'bert']:
        save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

        ner_data = NERData(models[model_name], tag_format=config['format'])
        ner_data.preprocess(datafile)

        train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(batch_size=32, train_frac=split[0], val_frac=split[1], dev_frac=split[2], seed=seed)
        classes = ner_data.classes

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
