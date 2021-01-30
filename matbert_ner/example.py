from models.bert_model import BertCRFNERModel
from utils.data import NERData
import os
import json

# datafile = "data/aunpmorph_annotations_fullparas.json"
datafile = "data/ner_annotations.json"
n_epochs = 128

device = "cuda"
model_names = ['scibert', 'matbert']

splits = {'_{}'.format(i): [0.1*i, 0.1, 0.1] for i in range(1, 9)}
for alias, split in splits.items():
    for model_name in model_names:
        if model_name == 'scibert':
            model = "allenai/scibert_scivocab_uncased"
            save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)
        if model_name == 'matbert':
            model = "/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased"
            save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

        ner_data = NERData(model)
        ner_data.preprocess(datafile)

        train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(train_frac=split[0], val_frac=split[1], dev_frac=split[2], batch_size=64)
        classes = ner_data.classes

        ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=1e-5)
        ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=save_dir)
