from models.bert_model import BertCRFNERModel
from utils.data import NERData
import os

datafile = "data/ner_annotations.json"
n_epochs = 1

device = "cuda"
model = "allenai/scibert_scivocab_cased"

ner_data = NERData(model)
ner_data.preprocess(datafile)

train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(val_frac=0.4,dev_frac=0.4)

classes = ner_data.classes

ner_model = BertCRFNERModel(modelname=model, classes = classes, device=device, lr=5e-5)


ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=os.getcwd())

print(ner_model.predict("The spherical nanoparticles were synthesized using an injection process in a cylindrical beaker."))