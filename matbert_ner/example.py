from models.bert_model import BertNER, BertCrfForNer
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup
from utils.data import InputExample, convert_examples_to_features, collate_fn
from utils.dataloader import load_data
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm
import numpy as np

batch_size = 20
n_epochs = 20

validation_split = 0.2
shuffle_dataset = True
device = "cuda"

# config = AutoConfig.from_pretrained("allenai/scibert_scivocab_cased")
config = AutoConfig.from_pretrained("/home/amalie/MatBERT/matbert_data/matbert-base-cased")

print(config.hidden_dropout_prob)
config.num_labels = 19

tokenizer = BertTokenizer.from_pretrained("/home/amalie/MatBERT/matbert_data/matbert-base-cased")
ner_model = BertCrfForNer(config).to(device)

datafile = "data/ner_annotations.json"

tensor_dataset = load_data(datafile, tokenizer)

dataset_size = len(tensor_dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
    num_workers=0, sampler=train_sampler)
valid_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
    num_workers=0, sampler=valid_sampler)


no_decay = ["bias", "LayerNorm.weight"]
bert_parameters = ner_model.bert.named_parameters()
classifier_parameters = ner_model.classifier.named_parameters()
bert_lr = 2e-5
classifier_lr = 5e-5
optimizer_grouped_parameters = [
    {"params": [p for n, p in bert_parameters if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.0,
     "lr": bert_lr},
    {"params": [p for n, p in bert_parameters if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0,
     "lr": bert_lr},

    {"params": [p for n, p in classifier_parameters if not any(nd in n for nd in no_decay)],
     "weight_decay": 0.0,
     "lr": classifier_lr},
    {"params": [p for n, p in classifier_parameters if any(nd in n for nd in no_decay)],
     "weight_decay": 0.0,
     "lr": classifier_lr}
]
optimizer = optim.AdamW(optimizer_grouped_parameters, 5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=n_epochs*len(train_dataloader)/(batch_size)
)

def accuracy(predicted, labels):
    predicted = torch.max(predicted,-1)[1]

    true = torch.where(labels > 0, labels, 0)
    predicted = torch.where(labels > 0, predicted, -1)

    acc = (true==predicted).sum().item()/torch.count_nonzero(true)
    return acc

for epoch in range(n_epochs):
    print("\n\n\nEpoch: " + str(epoch + 1))
    ner_model.train()

    for i, batch in enumerate(tqdm(train_dataloader)):
        
        inputs = {"input_ids": batch[0].to(device),
                  "attention_mask": batch[1].to(device),
                  "valid_mask": batch[2].to(device),
                  "labels": batch[4].to(device)}

        optimizer.zero_grad()
        loss, predicted = ner_model.forward(**inputs)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i%100 == 0:
            labels = inputs['labels']

            acc = accuracy(predicted, labels)

            print("loss: {}, acc: {}".format(torch.mean(loss).item(),acc.item()))

    ner_model.eval()
    val_loss = []
    val_pred = []
    val_label = []
    with torch.no_grad():
        for batch in valid_dataloader:

            inputs = {"input_ids": batch[0].to(device),
                      "attention_mask": batch[1].to(device),
                      "valid_mask": batch[2].to(device),
                      "labels": batch[4].to(device)}

            loss, pred = ner_model.forward(**inputs)
            val_loss.append(loss)
            val_pred.append(pred)
            val_label.append(inputs['labels'])
        val_loss = torch.stack(val_loss)
        val_pred = torch.cat(val_pred, dim=0)
        val_label = torch.cat(val_label, dim=0)
        val_acc = accuracy(val_pred, val_label)

    print("val loss: {}, val acc: {}".format(torch.mean(val_loss).item(),val_acc.item()))

