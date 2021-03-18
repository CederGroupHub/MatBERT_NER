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
raw_data = []
with open(datafile, 'r') as f:
    for l in f:
        raw_data.append(json.loads(l))
print(len(raw_data))
n_epochs = 1
batch_size = 20
seed = 94
device = "cuda"
model = '/home/amalie/MatBERT/matbert-base-uncased'
model = 'allenai/scibert_scivocab_uncased'
# model = '/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased'
save_dir = os.getcwd()+'/{}_results{}/'.format("scibert", "doc_emb")
# save_dir = None

ner_data = NERData(model)
ner_data.preprocess(datafile, split_on_sentences=True)

train_frac = 0.1
val_frac = 0.3
dev_frac = 1 - (train_frac + val_frac)
train_dataloader, val_dataloader, dev_dataloader = ner_data.create_dataloaders(batch_size=batch_size, train_frac=train_frac, val_frac=val_frac, dev_frac=dev_frac, seed=seed)

dataset_size = len(ner_data.dataset)
indices = list(range(dataset_size))
dev_split = int(np.floor(dev_frac * dataset_size))
val_split = int(np.floor(val_frac * dataset_size))+dev_split
np.random.seed(seed)
np.random.shuffle(indices)

dev_indices, val_indices = indices[:dev_split], indices[dev_split:val_split]

train_split = int(np.floor(train_frac * dataset_size))+val_split
train_indices = indices[val_split:train_split]

classes = ner_data.classes

def flatten(x):
    return [y for z in x for y in z]
ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=2e-4)
print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
# print(ner_model.model)

ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, dev_dataloader=dev_dataloader, save_dir=save_dir)

ner_model.model.eval()
all_logits = []
with torch.no_grad():
    for i, batch in enumerate(val_dataloader):
        inputs = {"input_ids": batch[0].to(device, non_blocking=True),
                  "attention_mask": batch[1].to(device, non_blocking=True),
                  "valid_mask": batch[2].to(device, non_blocking=True),
                  "labels": batch[4].to(device, non_blocking=True),
                  "return_logits": True}

        loss, predicted, logits = ner_model.model.forward(**inputs)
        logits = torch.where(batch[1].unsqueeze(-1).type(torch.bool), logits.detach().cpu().type(torch.double), float(0))
        all_logits.append(logits)
        
all_logits = torch.cat(all_logits)
all_logits = torch.max(all_logits,dim=-1)[0]
reduced_logits = []
for doc in all_logits:
    # logit_mean = doc[doc != 0].mean().numpy()
    logit_mean = doc.max().numpy()
    reduced_logits.append(float(logit_mean))

new_train_ids = np.argpartition(reduced_logits, len(train_indices))[:len(train_indices)]
new_train_ids = [val_indices[n] for n in new_train_ids]

old_train_indices = train_indices
old_val_indices = val_indices
train_indices = list(train_indices) + list(new_train_ids)
val_indices = [v for v in val_indices if v not in new_train_ids]
dev_indices = [v for v in dev_indices if v not in new_train_ids]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SequentialSampler(val_indices)
dev_sampler = SequentialSampler(dev_indices)

train_dataloader = DataLoader(ner_data.dataset, batch_size=batch_size,
    num_workers=0, sampler=train_sampler, pin_memory=True)

val_dataloader = DataLoader(ner_data.dataset, batch_size=batch_size,
    num_workers=0, sampler=val_sampler, pin_memory=True)

ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=2e-4)
ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, dev_dataloader=dev_dataloader, save_dir=save_dir)

np.random.shuffle(old_val_indices)
new_train_ids = old_val_indices[:len(old_train_indices)]
train_indices = list(old_train_indices) + list(new_train_ids)
val_indices = [v for v in old_val_indices if v not in train_indices]


train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SequentialSampler(val_indices)
dev_sampler = SequentialSampler(dev_indices)

train_dataloader = DataLoader(ner_data.dataset, batch_size=batch_size,
    num_workers=0, sampler=train_sampler, pin_memory=True)

val_dataloader = DataLoader(ner_data.dataset, batch_size=batch_size,
    num_workers=0, sampler=val_sampler, pin_memory=True)

ner_model = BertCRFNERModel(modelname=model, classes=classes, device=device, lr=2e-4)
ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, dev_dataloader=dev_dataloader, save_dir=save_dir)

# print(all_logits.shape)
# print(all_logits[all_logits != 0].shape)
# print(reduced_logits)