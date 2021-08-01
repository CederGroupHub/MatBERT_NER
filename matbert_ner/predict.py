import os


import torch
from utils.data import NERData
from matbert_ner.models.bert_model import BERTNER
from matbert_ner.models.model_trainer import NERTrainer


def predict(texts_tokenized, model_file, state_path, scheme="IOBES", batch_size=256, device="cpu", seed=None):
    split_dict = {'predict': 1.0}

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

    torch.device('cuda' if gpu else 'cpu')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    ner_data = NERData(model_file, scheme=scheme)
    ner_data.preprocess(texts_tokenized, split_dict, is_file=False, annotated=False, sentence_level=False, shuffle=False, seed=seed)
    ner_data.create_dataloaders(batch_size=batch_size, shuffle=False, seed=seed)
    bert_ner = BERTNER(model_file=model_file, classes=ner_data.classes, scheme=scheme, seed=seed)
    bert_ner_trainer = NERTrainer(bert_ner, device)
    annotations = bert_ner_trainer.predict(
        ner_data.dataloaders['predict'],
        original_data=ner_data.data['predict'],
        predict_path=None,
        state_path=state_path
    )

    return annotations



