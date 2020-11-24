from models.bert_model import BertNER
from transformers import BertTokenizer, AutoConfig
from utils.data import InputExample, convert_examples_to_features, collate_fn
from utils.dataloader import load_data
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

batch_size = 100

device = "cuda"

config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 19

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
ner_model = BertNER(config).to(device)

datafile = "data/ner_annotations.json"

tensor_dataset = load_data(datafile, tokenizer)
dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

optimizer = optim.AdamW(ner_model.parameters(), lr=0.01)


for epoch in range(4):
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
        predicted = torch.max(predicted,1)
        true = (inputs['labels']==predicted).sum().item()/batch_size

        if i%100 == 0:
            print(loss,true)
