import os

data_file = './data/solid_state.json'
model_file = '../../matbert-base-uncased'
save_dir = './matbert_solid_state_paragraph_iobes_crf_10_adamw_5_1_012_1e-04_1e-04_1e-03_exponential_256_80/'
state_path = save_dir+'best.pt'
scheme = 'IOBES'
split_dict = {'predict': 1.0}
batch_size = 10
sentence_level = False
seed = None
device = 'cpu'
if 'gpu' in device:
    gpu = True
    try:
        d, n = device.split(':')
    except:
        print('ValueError: Improper device format in command-line argument')
    device = 'cuda'
else:
    gpu = False
if gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(n)
import torch
from utils.data import NERData
from models.bert_model import BERTNER
from models.model_trainer import NERTrainer

torch.device('cuda' if gpu else 'cpu')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

ner_data = NERData(model_file, scheme=scheme)
ner_data.preprocess(data_file, split_dict, is_file=True, sentence_level=False, shuffle=False, seed=seed)
ner_data.create_dataloaders(batch_size=batch_size, shuffle=False, seed=seed)
bert_ner = BERTNER(model_file=model_file, classes=ner_data.classes, scheme=scheme, seed=seed)
bert_ner_trainer = NERTrainer(bert_ner, device)
annotations = bert_ner_trainer.predict(ner_data.dataloaders['predict'], predict_path=save_dir+'predict_{}.pt'.format(data_file.split('/')[-1].replace('.json', '')), state_path=state_path)
with open(save_dir+'predictions_{}.txt'.format(data_file.split('/')[-1].replace('.json', '')), 'w') as f:
    for entry in annotations:
        f.write(160*'='+'\n')
        for sentence in entry['tokens']:
            f.write(160*'-'+'\n')
            for word in sentence:
                f.write('{:<40}{:<40}\n'.format(word['text'], word['annotation']))
            f.write(160*'-'+'\n')
        f.write(160*'-'+'\n')
        for entity_type in entry['entities'].keys():
            f.write('{:<20}{}\n'.format(entity_type, ', '.join(entry['entities'][entity_type])))
        f.write(160*'-'+'\n')
        f.write(160*'='+'\n')
