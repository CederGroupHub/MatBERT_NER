from models.bert_model import BertNER, BertCrfForNer
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup
from utils.data import InputExample, convert_examples_to_features, collate_fn
from utils.dataloader import load_data
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from itertools import product

val_frac = 0.02
dev_frac = 0.02
shuffle_dataset = True

device = "cuda"

models = reversed([("scibert-cased", "allenai/scibert_scivocab_cased"), ("matbert-cased","/home/amalie/MatBERT_NER/matbert_ner/matbert-base-cased"),("scibert-uncased", "allenai/scibert_scivocab_uncased"), ("matbert-uncased","/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased")])
lrs = [5e-5, 1e-4, 1e-5, 2e-5, 8e-5]
epochs = [10, 20]

results_file = "ner_results_v2.csv"
for n_epochs, lr, model in product(epochs, lrs, models):
    print("{}-{}-{}".format(model[0],lr,n_epochs))
    config = AutoConfig.from_pretrained(model[1])

    config.num_labels = 19

    tokenizer = BertTokenizer.from_pretrained(model[1])

    datafile = "data/ner_annotations.json"

    tensor_dataset = load_data(datafile, tokenizer)

    batch_size = 20

    dataset_size = len(tensor_dataset)
    indices = list(range(dataset_size))
    dev_split = int(np.floor(dev_frac * dataset_size))
    val_split = int(np.floor(val_frac * dataset_size))+dev_split
    if shuffle_dataset :
        np.random.seed(1000)
        np.random.shuffle(indices)
    dev_indices, val_indices, train_indices = indices[:dev_split], indices[dev_split:val_split], indices[val_split:]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    dev_sampler = SubsetRandomSampler(dev_indices)

    train_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
        num_workers=0, sampler=train_sampler)
    val_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
        num_workers=0, sampler=val_sampler)
    dev_dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
        num_workers=0, sampler=dev_sampler)


    ner_model = BertCrfForNer(config).to(device)

    no_decay = ["bias", "LayerNorm.weight"]
    bert_parameters = ner_model.bert.named_parameters()
    classifier_parameters = ner_model.classifier.named_parameters()
    bert_lr = lr
    classifier_lr = lr
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
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=n_epochs*len(train_dataloader)
    )

    def accuracy(predicted, labels):
        predicted = torch.max(predicted,-1)[1]

        true = torch.where(labels > 0, labels, 0)
        predicted = torch.where(labels > 0, predicted, -1)

        acc = (true==predicted).sum().item()/torch.count_nonzero(true)
        return acc

    val_loss_best = 500000
    save_path = "{}_{}_{}_best.pt".format(model[0], lr, n_epochs)
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
            for batch in val_dataloader:

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

        if torch.mean(val_loss).item() < val_loss_best:
            torch.save(ner_model.state_dict(), save_path)

        print("val loss: {}, val acc: {}".format(torch.mean(val_loss).item(),val_acc.item()))

    ner_model.load_state_dict(torch.load(save_path))
    ner_model.eval()
    val_loss = []
    val_pred = []
    val_label = []

    with torch.no_grad():
        for batch in dev_dataloader:

            inputs = {"input_ids": batch[0].to(device),
                      "attention_mask": batch[1].to(device),
                      "valid_mask": batch[2].to(device),
                      "labels": batch[4].to(device)}

            loss, pred = ner_model.forward(**inputs)
            val_loss.append(loss)
            val_pred.append(pred)
            val_label.append(inputs['labels'])
        val_loss = torch.mean(torch.stack(val_loss)).item()
        val_pred = torch.cat(val_pred, dim=0)
        val_label = torch.cat(val_label, dim=0)
        val_acc = accuracy(val_pred, val_label).item()

    with open(results_file,"a+") as f:
        f.write("{},{},{},{},{}\n".format(model[0],lr,n_epochs,val_loss,val_acc))
