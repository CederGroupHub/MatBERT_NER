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

seed = 128
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

datafiles = {'ner_annotations': 'data/ner_annotations.json',
             'doping': 'data/doping.json',
             'impurityphase': 'data/impurityphase_fullparas.json',
             'aunpmorph': 'data/aunpmorph_annotations_fullparas.json'}

n_epochs = 8
lr = 2e-4
splits = 30
val_frac = 0.10
batch_size=32
device = "cuda"
models = {'bert': 'bert-base-uncased',
          'scibert': 'allenai/scibert_scivocab_uncased',
          'matbert': '/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased'}

data = 'doping'
#data = 'ner_annotations'
configs = {}
#configs['_{}_reduced_full_crf_iobes_{}'.format(data, seed)] = {'full_finetuning': True,
#                                                               'format': 'IOBES'}
for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    configs['_{}_full_crf_iob2_{}_reduced_{})'.format(data, seed, frac)] = {'full_finetuning': True,
                                                                            'format': 'IOB2', "frac": frac}

# configs['_{}_reduced_shallow_crf_iobes_{}'.format(data, seed)] = {'full_finetuning': False,
#                                                                   'format': 'IOBES'}
# configs['_{}_reduced_shallow_crf_iob2_{}'.format(data, seed)] = {'full_finetuning': False,
                                                                 # 'format': 'IOB2'}
from itertools import product
for ac, model_name in product(configs.items(),["matbert", "scibert", "bert"]):
    alias, config = ac
    print(alias, model_name)
    #model_name = 'bert'
    save_dir = os.getcwd()+'/{}_results{}/'.format(model_name, alias)

    ner_data = NERData(models[model_name], tag_format=config['format'])
    ner_data.preprocess(datafiles[data])

    frac = config['frac']
    dataloaders = ner_data.create_kfold_dataloaders(batch_size=batch_size, val_frac=val_frac, seed=seed, splits=splits, frac=frac)
    classes = ner_data.classes

    prediction_tags = []
    label_tags = []
    loss = []
    i = 0
    for train_dataloader, val_dataloader, dev_dataloader in dataloaders:
        print(i)
        i += 1
        #print(len(train_dataloader)*batch_size, len(val_dataloader)*batch_size, len(dev_dataloader)*batch_size)
        ner_model = BertCRFNERModel(modelname=models[model_name], classes=classes, tag_format=config['format'], device=device, lr=lr)
        #print('{} classes: {}'.format(len(ner_model.classes),' '.join(ner_model.classes)))
        #print(ner_model.model)
        ner_model.train(train_dataloader, n_epochs=n_epochs, val_dataloader=val_dataloader, dev_dataloader=dev_dataloader, save_dir=save_dir, full_finetuning=config['full_finetuning'])

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
    #prediction_tags = [x for k in prediction_tags for x in k]
    #label_tags = [x for k in label_tags for x in k]
    #loss = torch.cat(loss)

    metric_mode = 'strict'
    metric_scheme = IOB2
    metrics = dict()
    metrics['loss'] = [float(np.mean(l.detach().cpu().numpy())) for l in loss]
    metrics['accuracy_score'] = [float(accuracy_score(lt, pt)) for lt,pt in zip(label_tags, prediction_tags)]
    metrics['micro_precision_score'] = [float(precision_score(lt, pt, mode=metric_mode, scheme=metric_scheme)) for lt,pt in zip(label_tags, prediction_tags)]
    metrics['micro_recall_score'] = [float(recall_score(lt, pt, mode=metric_mode, scheme=metric_scheme)) for lt,pt in zip(label_tags, prediction_tags)]
    metrics['micro_f1_score'] = [float(f1_score(lt, pt, mode=metric_mode, scheme=metric_scheme)) for lt,pt in zip(label_tags, prediction_tags)]

    metrics['macro_precision_score'] = [float(precision_score(lt, pt, mode=metric_mode, scheme=metric_scheme, average='macro')) for lt,pt in zip(label_tags, prediction_tags)]
    metrics['macro_recall_score'] = [float(recall_score(lt, pt, mode=metric_mode, scheme=metric_scheme, average='macro')) for lt,pt in zip(label_tags, prediction_tags)]
    metrics['macro_f1_score'] = [float(f1_score(lt, pt, mode=metric_mode, scheme=metric_scheme, average='macro')) for lt,pt in zip(label_tags, prediction_tags)]

    print(metrics)

    mean_metrics = {k:np.mean(v) for k,v in metrics.items()}
    print(mean_metrics)
    with open(os.path.join(save_dir,"metrics.json"),'w') as f:
        f.write(json.dumps(metrics))
