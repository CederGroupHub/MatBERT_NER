from models.bert_model import BertNER
from transformers import BertTokenizer, AutoConfig
from utils.data import InputExample, convert_examples_to_features
import torch

device = "cuda"

config = AutoConfig.from_pretrained("bert-base-uncased")
example = [("22","B-Chemical"),("-","I-Chemical"),("oxacalcitriol","I-Chemical"),("suppresses","O")]
label_dict = {"O":0, "B-Chemical":1, "I-Chemical":2}
example = InputExample(0,[e[0] for e in example], [e[1] for e in example])

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
ner_model = BertNER(config).to(device)

feats = convert_examples_to_features(
        [example],
        ["O", "B-Chemical", "I-Chemical"],
        10,
        tokenizer,
)

input_ids = feats[0].input_ids
input_mask = feats[0].input_mask
valid_mask = feats[0].valid_mask
segment_ids = feats[0].segment_ids
label_ids = feats[0].label_ids


inputs = {"input_ids": torch.tensor(input_ids).unsqueeze(0).to(device),
          "attention_mask": torch.tensor(input_mask).unsqueeze(0).to(device),
          "valid_mask": torch.tensor(valid_mask).unsqueeze(0).to(device),
          "labels": torch.tensor(label_ids).unsqueeze(0).to(device), }

print(inputs)

print(ner_model.forward(**inputs))