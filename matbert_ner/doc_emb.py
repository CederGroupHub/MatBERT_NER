import torch
from models.bert_model import BertCRFNERModel
from models.bilstm_model import BiLSTMNERModel
from utils.data import NERData
import os
import glob
import json
import numpy as np

# datafile = 'data/impurityphase_fullparas.json'
datafile = 'data/aunpmorph_annotations_fullparas.json'
#datafile = "data/bc5dr.json"
# datafile = "data/ner_annotations.json"
n_epochs = 2
full_finetuning = True

device = "cuda"
model = 'allenai/scibert_scivocab_uncased'
save_dir = os.getcwd()+'/{}_results{}/'.format("scibert", "doc_emb")
# save_dir = None

ner_data = NERData(model)
ner_data.preprocess(datafile)

train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(train_frac=0.8, val_frac=0.01, dev_frac=0.19, batch_size=20)
classes = ner_data.classes

ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=2e-4)
print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
print(ner_model.model)
dev_doc_embs = []
ner_model.model.eval()
with torch.no_grad():
    for batch in dev_dataloader:
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
        }
        doc_embs = ner_model.document_embeddings(**inputs)
        dev_doc_embs.append(doc_embs.detach().cpu())
dev_doc_embs = torch.cat(dev_doc_embs)
train_doc_embs = []
ner_model.model.eval()
with torch.no_grad():
    for batch in train_dataloader:
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
        }
        doc_embs = ner_model.document_embeddings(**inputs)
        train_doc_embs.append(doc_embs.detach().cpu())
train_doc_embs = torch.cat(train_doc_embs)
mean_train_doc_emb = torch.mean(train_doc_embs, dim=0)

ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=save_dir, full_finetuning=full_finetuning)

ner_model.model.eval()
dev_losses = []
with torch.no_grad():
    for batch in dev_dataloader:
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
            "valid_mask": batch[2].to(device),
            "labels": batch[4].to(device),
            "reduction": "none"
        }
        loss, _ = ner_model.model.forward(**inputs)
        dev_losses.append(loss.detach().cpu())

dev_losses = torch.cat(dev_losses)

dev_doc_norms = torch.linalg.norm(dev_doc_embs,dim=1)
train_doc_norms = torch.linalg.norm(train_doc_embs,dim=1)

norms = torch.einsum("i,j->ij", dev_doc_norms, train_doc_norms)

dev_doc_dots = torch.einsum("ij,kj->ik", dev_doc_embs, train_doc_embs)

dev_doc_cosine_sims = torch.div(dev_doc_dots, norms) 

knn = 5

dev_doc_dots = torch.sort(dev_doc_dots,descending=True)[0][:,:5]
dev_doc_dots = torch.mean(dev_doc_dots,dim=1)   

dev_doc_cosine_sims = torch.sort(dev_doc_cosine_sims,descending=True)[0][:,:5]
dev_doc_cosine_sims = torch.mean(dev_doc_cosine_sims,dim=1)   

# dev_doc_dots = torch.max(dev_doc_dots, dim=1)[0]
# dev_doc_cosine_sims = torch.max(dev_doc_cosine_sims, dim=1)[0]

# dev_doc_dots = torch.einsum("ij,j->i", dev_doc_embs, mean_train_doc_emb)

# dev_doc_cosine_sims = torch.nn.CosineSimilarity(dim=1)(dev_doc_embs, torch.unsqueeze(mean_train_doc_emb, dim=0))


with open("dev_losses_dot.csv",'w') as f:
    for l,d,c in zip(dev_losses.cpu().numpy(), dev_doc_dots.cpu().numpy(), dev_doc_cosine_sims.cpu().numpy()):
        f.write("{},{},{}\n".format(l,d,c))