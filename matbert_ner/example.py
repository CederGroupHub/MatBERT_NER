from models.bert_model import BertCRFNERModel
from utils.data import NERData
import os

datafile = "data/ner_annotations.json"
n_epochs = 15

device = "cuda"
model = "allenai/scibert_scivocab_cased"

ner_data = NERData(model)
ner_data.preprocess(datafile)

train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders()

classes = ner_data.classes

ner_model = BertCRFNERModel(model=model, classes = classes, device=device, lr=5e-5)


ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=os.getcwd())