from models.bert_model import BertNER
from transformers import BertTokenizer, AutoConfig, get_linear_schedule_with_warmup
from utils.data import InputExample, convert_examples_to_features, collate_fn
from utils.dataloader import load_data
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

batch_size = 20
n_epochs = 5

device = "cuda"

config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 19

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
ner_model = BertNER(config).to(device)

datafile = "data/ner_annotations.json"

tensor_dataset = load_data(datafile, tokenizer)
dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)


no_decay = ["bias", "LayerNorm.weight"]
bert_parameters = ner_model.bert.named_parameters()
classifier_parameters = ner_model.classifier.named_parameters()
bert_lr = 5e-5
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
    optimizer, num_warmup_steps=0, num_training_steps=n_epochs*len(dataloader)/(batch_size)
)


for epoch in range(n_epochs):
    print("\n\n\nEpoch: " + str(epoch + 1))
    for i, batch in enumerate(tqdm(dataloader)):
        
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
            predicted = torch.max(predicted,-1)[1]

            true = torch.where(labels > 0, labels, 0)
            predicted = torch.where(labels > 0, predicted, -1)

            acc = (true==predicted).sum().item()/torch.count_nonzero(true)

            print("loss: {}, acc: {}".format(loss.item(),acc.item()))
