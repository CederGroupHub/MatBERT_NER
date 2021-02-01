from models.bert_model import BertCRFNERModel
from utils.data import NERData
import os
import glob
import json

# datafile = "data/aunpmorph_annotations_fullparas.json"
datafile = "data/ner_annotations.json"
n_epochs = 128

device = "cuda"
model_names = ['bert']

splits = {'': [0.5, 0.25, 0.25]}
for alias, split in splits.items():
    for model_name in model_names:
        if model_name == 'bert':
            model = 'bert-base-uncased'
        if model_name == 'scibert':
            model = "allenai/scibert_scivocab_uncased"
        if model_name == 'matbert':
            model = "/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased"
        save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

        ner_data = NERData(model)
        ner_data.preprocess(datafile)

        train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(train_frac=split[0], val_frac=split[1], dev_frac=split[2], batch_size=64)
        classes = ner_data.classes

        ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=1e-5)
        ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=save_dir)

        fs = glob.glob(save_dir+'epoch_*pt')
        for f in fs:
            try:
                os.remove(f)
            except:
                print('error while deleting file: {}'.format(f))
