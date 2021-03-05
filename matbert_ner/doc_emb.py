import torch
from models.bert_model import BertCRFNERModel
from models.bilstm_model import BiLSTMNERModel
from utils.data import NERData
import os
import glob
import json
import numpy as np
from itertools import product
from torch.utils.data import DataLoader, SubsetRandomSampler, TensorDataset, SequentialSampler

# datafile = 'data/impurityphase_fullparas.json'
# datafile = 'data/aunpmorph_annotations_fullparas.json'
#datafile = "data/bc5dr.json"
datafile = "data/ner_annotations.json"
n_epochs = 4
full_finetuning = True
batch_size = 20

device = "cuda"
model = 'allenai/scibert_scivocab_uncased'
save_dir = os.getcwd()+'/{}_results{}/'.format("scibert", "doc_emb")
# save_dir = None

ner_data = NERData(model)
ner_data.preprocess(datafile)

classes = ner_data.classes

def flatten(x):
    return [y for z in x for y in z]
ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=2e-4)
print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
print(ner_model.model)
ner_model.model.eval()

train_frac = 0.2
all_dataloader = DataLoader(ner_data.dataset, batch_size=batch_size,
            num_workers=0, pin_memory=True)
train_set_size = int(train_frac*len(all_dataloader)*batch_size)
doc_embs = []
with torch.no_grad():
    for batch in all_dataloader:
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
        }
        doc_emb = ner_model.document_embeddings(**inputs)
        doc_embs.append(doc_emb.detach().cpu())

doc_embs = torch.cat(doc_embs)

dists = torch.zeros((doc_embs.shape[0],doc_embs.shape[0]), dtype=torch.double)

for i,j in product(range(doc_embs.shape[0]),range(doc_embs.shape[0])):
    dists[i,j] = torch.linalg.norm(doc_embs[i,:] - doc_embs[j,:],dtype=torch.double)

train_mask = torch.full((doc_embs.shape[0],doc_embs.shape[0]), False, dtype=torch.bool)

train_indices = [0]
last_train_index = train_indices[-1]
while len(train_indices) < train_set_size:
    train_mask[:,last_train_index] = True
    dists[last_train_index,:] = float(1e6)

    masked_dists = torch.where(train_mask, dists, float(1e6))
    masked_mins = torch.min(masked_dists, dim=1)[0]
    last_train_index = torch.min(masked_mins,dim=0)[1]
    train_indices.append(int(last_train_index))

dev_indices = [x for x in range(len(all_dataloader)*batch_size) if not x in train_indices]

print(len(train_indices), len(dev_indices))
train_sampler = SubsetRandomSampler(train_indices)
dev_sampler = SequentialSampler(dev_indices)

train_dataloader = DataLoader(ner_data.dataset, batch_size=batch_size,
    num_workers=0, sampler=train_sampler, pin_memory=True)

dev_dataloader = DataLoader(ner_data.dataset, batch_size=batch_size,
    num_workers=0, sampler=dev_sampler, pin_memory=True)

# train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(train_frac=0.7, val_frac=0.01, dev_frac=0.29 , batch_size=20)

train_doc_embs = []
ner_model.model.eval()
train_vocab = set()

with torch.no_grad():
    for batch in train_dataloader:
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
        }
        train_vocab.update(flatten(batch[0].detach().cpu().tolist()))
        doc_embs = ner_model.document_embeddings(**inputs)
        train_doc_embs.append(doc_embs.detach().cpu())
train_doc_embs = torch.cat(train_doc_embs)
mean_train_doc_emb = torch.mean(train_doc_embs, dim=0)


dev_doc_embs = []
dev_unique_tokens = []
with torch.no_grad():
    for batch in dev_dataloader:
        inputs = {
            "input_ids": batch[0].to(device),
            "attention_mask": batch[1].to(device),
        }
        doc_embs = ner_model.document_embeddings(**inputs)
        dev_doc_embs.append(doc_embs.detach().cpu())
        tokens = set(flatten(batch[0].detach().cpu().tolist()))
        unique_tokens = tokens - tokens.intersection(train_vocab)
        dev_unique_tokens.append(len(unique_tokens))
dev_doc_embs = torch.cat(dev_doc_embs)

# , val_dataloader=val_dataloader
ner_model.train(train_dataloader, val_dataloader=dev_dataloader, n_epochs=n_epochs, save_dir=save_dir, full_finetuning=full_finetuning)

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

dev_dists = torch.zeros((dev_doc_embs.shape[0],train_doc_embs.shape[0]))
for i,j in product(range(dev_doc_embs.shape[0]),range(train_doc_embs.shape[0])):
    dev_dists[i,j] = torch.linalg.norm(dev_doc_embs[i,:] - train_doc_embs[j,:])
# dev_dists = dev_doc_embs.unsqueeze(dim=1).repeat(1,train_doc_embs.shape[0],1)
# train_dists = train_doc_embs.unsqueeze(dim=0).repeat(dev_doc_embs.shape[0],1,1)
# dev_dists = dev_dists - train_dists
# dev_dists = torch.linalg.norm(dev_dists,dim=-1)

dev_dists_mins = torch.min(dev_dists,dim=1)[0]


# dev_doc_norms = torch.linalg.norm(dev_doc_embs,dim=1)
# train_doc_norms = torch.linalg.norm(train_doc_embs,dim=1)

# norms = torch.einsum("i,j->ij", dev_doc_norms, train_doc_norms)

dev_doc_dots = torch.einsum("ij,kj->ik", dev_doc_embs, train_doc_embs)

# dev_doc_cosine_sims = torch.div(dev_doc_dots, norms) 

knn = 2

dev_doc_dots = torch.sort(dev_doc_dots,descending=True)[0][:,:knn]
dev_doc_dots = torch.mean(dev_doc_dots,dim=1)   

dev_dists_knn = torch.sort(dev_dists,descending=False)[0][:,:knn]
dev_dists_knn = torch.mean(dev_dists_knn,dim=1)

# dev_doc_cosine_sims = torch.sort(dev_doc_cosine_sims,descending=True)[0][:,:5]
# dev_doc_cosine_sims = torch.mean(dev_doc_cosine_sims,dim=1)   

# dev_doc_dots = torch.max(dev_doc_dots, dim=1)[0]
# dev_doc_cosine_sims = torch.max(dev_doc_cosine_sims, dim=1)[0]

# dev_doc_dots = torch.einsum("ij,j->i", dev_doc_embs, mean_train_doc_emb)

# dev_doc_cosine_sims = torch.nn.CosineSimilarity(dim=1)(dev_doc_embs, torch.unsqueeze(mean_train_doc_emb, dim=0))


# with open("dev_losses_dot.csv",'w') as f:
    # for l,d,c,k,t in zip(dev_losses.cpu().numpy(), dev_doc_dots.cpu().numpy(), dev_dists_mins.cpu().numpy(), dev_dists_knn.cpu().numpy(), dev_unique_tokens):
        # f.write("{},{},{},{}\n".format(l,d,c,k,t))