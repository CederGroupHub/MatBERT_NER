import torch
from utils.data import NERData
from models.bert_model import BERTNER
from models.model_trainer import NERTrainer

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

ner_data = NERData(model_file, scheme=scheme)
ner_data.preprocess(data_file, split_dict, is_file=True, sentence_level=False, shuffle=False, seed=seed)
ner_data.create_dataloaders(batch_size=batch_size, shuffle=False, seed=seed)
bert_ner = BERTNER(model_file=model_file, classes=ner_data.classes, scheme=scheme, seed=seed)
bert_ner_trainer = NERTrainer(bert_ner, device)
annotations = bert_ner_trainer.predict(ner_data.dataloaders['predict'], predict_path=save_dir+'predict.pt', state_path=state_path)
with open(save_dir+'predictions_full.txt', 'w') as f:
    for entry in annotations:
        f.write(80*'='+'\n')
        for sentence in entry:
            f.write(80*'-'+'\n')
            for word in sentence:
                f.write('{:<20}{:<20}\n'.format(word['text'], word['annotation']))
            f.write(80*'-'+'\n')
        f.write(80*'='+'\n')