from models.bert_model import BertCRFNERModel
from utils.data import NERData
import os
import json

# datafile = "data/aunpmorph_annotations_fullparas.json"
datafile = "data/ner_annotations.json"
n_epochs = 4

device = "cuda"
model = "allenai/scibert_scivocab_uncased"
# model = "/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased"

ner_data = NERData(model)
ner_data.preprocess(datafile)

train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(val_frac=0.25, dev_frac=0.25, batch_size=64)
classes = ner_data.classes

ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=1e-5)
ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=os.getcwd())

# print(ner_model.predict("The spherical nanoparticles were synthesized using an injection process in a cylindrical beaker."))
