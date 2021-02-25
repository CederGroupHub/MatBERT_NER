from models.bert_model import BertCRFNERModel
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
# datafile = 'data/ner_annotations.json'
datafile = 'data/doping.json'

n_epochs = 16
crf_penalties = True

device = "cuda"
models = {'bert': 'bert-base-uncased',
          'scibert': 'allenai/scibert_scivocab_uncased',
          'matbert': '/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased'}

configs = {'_doping_iob2_full': {'tag_format': 'IOB2', 'split': [0.8, 0.1, 0.1], 'full_finetuning': True},
           '_doping_iob2_shallow': {'tag_format': 'IOB2', 'split': [0.8, 0.1, 0.1], 'full_finetuning': False},
           '_doping_bioes_full': {'tag_format': 'BIOES', 'split': [0.8, 0.1, 0.1], 'full_finetuning': True},
           '_doping_bioes_shallow': {'tag_format': 'BIOES', 'split': [0.8, 0.1, 0.1], 'full_finetuning': False}}
for alias, config in configs.items():
    tag_format = config['tag_format']
    split = config['split']
    full_finetuning = config['full_finetuning']
    for model_name in ['matbert', 'scibert', 'bert']:
        save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

        ner_data = NERData(models[model_name], tag_format=tag_format)
        ner_data.preprocess(datafile)

        train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders( batch_size=32, train_frac=split[0], val_frac=split[1], dev_frac=split[2], seed=seed)
        classes = ner_data.classes

        ner_model = BertCRFNERModel(modelname=models[model_name], classes=classes, tag_format=tag_format, crf_penalties=crf_penalties, device=device, lr=2e-4)
        print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
        print(ner_model.model)
        ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=save_dir, full_finetuning=full_finetuning)

        fs = glob.glob(save_dir+'epoch_*pt')
        for f in fs:
            try:
                os.remove(f)
            except:
                print('error while deleting file: {}'.format(f))
