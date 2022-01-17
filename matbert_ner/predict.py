import os
import json
import torch
from matbert_ner.utils.data import NERData
from matbert_ner.models.bert_model import BERTNER
from matbert_ner.models.model_trainer import NERTrainer


def predict(texts, is_file, model_file, state_path, predict_path=None, return_full_dict=False, scheme="IOBES", batch_size=256, device="cpu", seed=None):
    """
    Predict labels for texts. Please limit input to 512 tokens or less.

    Args:
        texts ([str]): JSON filename, list of JSON entries, or list of string texts to predict labels for. Untokenized text will be tokenized interally with
            the Materials Tokenizer.
        is_file (bool): Toggle for whether the texts are a JSON file or list of JSON entries/strings
        model_file (str): Path to BERT model file.
        state_path (str): Path to model state for NER task, fine tuned for specific task (e.g., gold nanoparticles).
        predict_path (str): Name of output file
        return_full_dict (bool): Toggle for returning the full JSON entry or just the summarized entities detected by the model
        scheme (str): IOBES or IOB2.
        batch_size (int): Number of samples to predict in one batch pass.
        device (str): Select 'cpu', 'gpu', or torch specific logic for running on multiple GPUs.
        seed (int, None): Seed for prediction.

    Returns:
        ([dict]): dictionaries of tokens and label annotations

    """
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
    ner_data.preprocess(texts, split_dict, is_file=is_file, annotated=False, sentence_level=False, shuffle=False, seed=seed)
    ner_data.create_dataloaders(batch_size=batch_size, shuffle=False, seed=seed)
    bert_ner = BERTNER(model_file=model_file, classes=ner_data.classes, scheme=scheme, seed=seed)
    bert_ner_trainer = NERTrainer(bert_ner, device)
    annotations = bert_ner_trainer.predict(ner_data.dataloaders['predict'],
                                           original_data=ner_data.data['predict'],
                                           state_path=state_path,
                                           predict_path=predict_path,
                                           return_full_dict=return_full_dict)
    return annotations
