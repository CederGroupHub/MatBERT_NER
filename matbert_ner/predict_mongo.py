import os
import pymongo
import itertools
import json
from datetime import datetime
import numpy as np
from tqdm import tqdm


def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

model_file = '../../matbert-base-uncased'
model_reference = 'matbert_solid_state_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_100'
save_dir = './{}/'.format(model_reference)
state_path = save_dir+'best.pt'
scheme = 'IOBES'
split_dict = {'predict': 1.0}
fetch_batch_size = 500
sentence_level = False
seed = None
device = 'gpu:0'
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

print('Preparing Mongo Client')
print(100*'=')
client = pymongo.MongoClient('mongodb03.nersc.gov', username=os.environ['MATSCHOLAR_DEV_USER'], password=os.environ['MATSCHOLAR_DEV_PASS'], authSource='matscholar_dev')
db = client['matscholar_dev']
print('Mongo Client Initialized')
print(100*'=')

dois = [d['meta']['doi'] for d in db.matbert_ner_entries_walkernr_test_v3.find()]

i = 0
ner_data = NERData(model_file, scheme=scheme)
for entries in grouper(fetch_batch_size, db.entries.find({'doi': {'$nin': dois}})):
    try:
        entries_clean = [{'meta': {'doi': entry['doi'], 'par': 0}, 'text': '{}. {}'.format(entry['title'], entry['abstract'])} for entry in entries]
        ner_data.preprocess(entries_clean, split_dict, is_file=False, annotated=False, sentence_level=False, shuffle=False, seed=seed)
        ner_data.create_dataloaders(batch_size=int(np.ceil(len(ner_data.data['predict'])/4)), shuffle=False, seed=seed)
        if i ==0:
            bert_ner = BERTNER(model_file=model_file, classes=ner_data.classes, scheme=scheme, seed=seed)
            bert_ner_trainer = NERTrainer(bert_ner, device)
            bert_ner_trainer.load_state(state_path=state_path, optimizer=False)
            labels = list(set(ner_data.classes))
        annotations = bert_ner_trainer.predict(ner_data.dataloaders['predict'], original_data=ner_data.data['predict'])
        for entry, annotation in tqdm(zip(entries_clean, annotations), desc='| updating entry user/model/date stamps |'):
            entry.update({key: annotation[key] for key in annotation.keys() if key != 'id'})
            entry.update({'user': 'walkernr', 'model': model_reference, 'date': datetime.now().strftime('%Y:%m:%d:%H:%M:%S')})
        db.matbert_ner_entries_walkernr_test_v3.insert_many(entries_clean)
        print(100*'=')
        print('Entries Written to DB')
        print(100*'=')
        i += 1
    except:
        print('Fetched Batch Failed')