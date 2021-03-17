from models.bert_model import BertCRFNERModel
from models.bilstm_model import BiLSTMNERModel
from utils.data import NERData
import os
import glob
import json
import torch
import numpy as np
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
from seqeval.scheme import IOB1, IOB2, IOBES

seed = 256
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

datafiles = {'ner_annotations': 'data/ner_annotations.json',
             'doping': 'data/doping.json',
             'impurityphase': 'data/impurityphase_fullparas.json',
             'aunpmorph': 'data/aunpmorph_annotations_fullparas.json'}

n_epochs = 1
lr = 2e-4
splits = 2
val_frac = 0.2

device = "cuda"
models = {'bert': 'bert-base-uncased',
          'scibert': 'allenai/scibert_scivocab_uncased',
          'matbert': '/home/amalie/MatBERT/matbert-base-uncased'}

data = 'aunpmorph'
configs = {}
configs['_{}_reduced_full_crf_iobes_{}'.format(data, seed)] = {'full_finetuning': True,
                                                               'format': 'IOBES'}
# configs['_{}_reduced_full_crf_iob2_{}'.format(data, seed)] = {'full_finetuning': True,
#                                                               'format': 'IOB2'}

# configs['_{}_reduced_shallow_crf_iobes_{}'.format(data, seed)] = {'full_finetuning': False,
#                                                                   'format': 'IOBES'}
# configs['_{}_reduced_shallow_crf_iob2_{}'.format(data, seed)] = {'full_finetuning': False,
                                                                 # 'format': 'IOB2'}

for alias, config in configs.items():
    model_name = 'matbert'
    save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

    ner_data = NERData(models[model_name], tag_format=config['format'])
    ner_data.preprocess(datafiles[data])

    dataloaders = ner_data.create_kfold_dataloaders(batch_size=32, val_frac=val_frac, seed=seed, splits=splits)
    classes = ner_data.classes

    prediction_tags = []
    label_tags = []
    loss = []
    for train_dataloader, val_dataloader, dev_dataloader in dataloaders:

        ner_model = BertCRFNERModel(modelname=models[model_name], classes=classes, tag_format=config['format'], device=device, lr=lr)
        print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
        print(ner_model.model)
        ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, save_dir=save_dir, full_finetuning=config['full_finetuning'])

        _, pt, lt, l = ner_model.predict(dev_dataloader, tok_dataset=ner_data.load_from_file(datafiles[data]), return_tags=True, labels=classes)

        prediction_tags.append(pt)
        label_tags.append(lt)
        loss.append(l)
        fs = glob.glob(save_dir+'epoch_*pt')
        for f in fs:
            try:
                os.remove(f)
            except:
                print('error while deleting file: {}'.format(f))
    prediction_tags = [x for k in prediction_tags for x in k]
    label_tags = [x for k in label_tags for x in k]
    loss = torch.cat(loss)

    metric_mode = 'strict'
    metric_scheme = IOBES
    metrics = dict()
    metrics['loss'] = np.mean(loss.detach().cpu().numpy())
    metrics['accuracy_score'] = accuracy_score(label_tags, prediction_tags)
    metrics['precision_score'] = precision_score(label_tags, prediction_tags, mode=metric_mode, scheme=metric_scheme)
    metrics['recall_score'] = recall_score(label_tags, prediction_tags, mode=metric_mode, scheme=metric_scheme)
    metrics['f1_score'] = f1_score(label_tags, prediction_tags, mode=metric_mode, scheme=metric_scheme)

    print(metrics)