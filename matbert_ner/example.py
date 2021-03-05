from models.bert_model import BertCRFNERModel
from models.bilstm_model import BiLSTMNERModel
from utils.data import NERData
import os
import glob
import json

# datafile = 'data/impurityphase_fullparas.json'
datafile = 'data/aunpmorph_annotations_fullparas.json'
#datafile = "data/bc5dr.json"
#datafile = "data/ner_annotations.json"
n_epochs = 4
full_finetuning = True

device = "cuda"
models = {'bert': 'bert-base-uncased',
          'scibert': 'allenai/scibert_scivocab_uncased',
          'matbert': '/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased'}

splits = {'_80_10_10': [0.8, 0.1, 0.1]}
# splits = {}
for alias, split in splits.items():
    for model_name in ['scibert']:
        save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

        ner_data = NERData(models[model_name])
        ner_data.preprocess(datafile)

        train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(train_frac=split[0], val_frac=split[1], dev_frac=split[2], batch_size=32)
        classes = ner_data.classes

        ner_model = BertCRFNERModel(modelname=models[model_name], classes=classes, device=device, lr=2e-4)
        print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
        print(ner_model.model)
        ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=save_dir, full_finetuning=full_finetuning)

        fs = glob.glob(save_dir+'epoch_*pt')
        for f in fs:
            try:
                os.remove(f)
            except:
                print('error while deleting file: {}'.format(f))
