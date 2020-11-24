from models.bert_model import BertNER
from transformers import BertTokenizer, AutoConfig
from utils.data import InputExample, convert_examples_to_features, collate_fn
from utils.dataloader import load_data
import torch
from torch.utils.data import DataLoader

batch_size = 10

device = "cuda"

config = AutoConfig.from_pretrained("bert-base-uncased")
config.num_labels = 19

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
ner_model = BertNER(config).to(device)

datafile = "data/ner_annotations.json"

tensor_dataset = load_data(datafile, tokenizer)
dataloader = DataLoader(tensor_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

batch = dataloader._get_iterator().__next__()
inputs = {"input_ids": batch[0].to(device),
          "attention_mask": batch[1].to(device),
          "valid_mask": batch[2].to(device),
          "labels": batch[4].to(device)}

print(ner_model.forward(**inputs))