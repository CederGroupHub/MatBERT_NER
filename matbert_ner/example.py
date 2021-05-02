import os
import argparse
import glob
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', help='computation device for model (e.g. cpu, gpu:0, gpu:1)', type=str, default='gpu:0')
    parser.add_argument('-sd', '--seeds', help='comma-separated seeds for data shuffling and model initialization (e.g. 1,2,3 or 2,4,8)', type=str, default='256')
    parser.add_argument('-ts', '--tag_schemes', help='comma-separated tagging schemes to be considered (e.g. IOB1,IOB2,IOBES)', type=str, default='IOBES')
    parser.add_argument('-st', '--splits', help='comma-separated training splits to be considered, in percent (e.g. 80). test split will always be 10%% and the validation split will be 1/8 of the training split', type=str, default='80')
    parser.add_argument('-ds', '--datasets', help='comma-separated datasets to be considered (e.g. solid_state,doping)', type=str, default='solid_state')
    parser.add_argument('-ml', '--models', help='comma-separated models to be considered (e.g. matbert,scibert,bert)', type=str, default='matbert')
    parser.add_argument('-sl', '--sentence_level', help='switch for sentence-level learning instead of paragraph-level', action='store_true')
    parser.add_argument('-df', '--deep_finetuning', help='switch for finetuning of pre-trained parameters', action='store_true')
    parser.add_argument('-ne', '--n_epochs', help='number of training epochs', type=int, default=16)
    parser.add_argument('-lr', '--learning_rate', help='optimizer learning rate', type=float, default=2e-4)
    parser.add_argument('-km', '--keep_model', help='switch for saving the best model parameters to disk', action='store_true')
    args = parser.parse_args()
    return args.device, args.seeds, args.tag_schemes, args.splits, args.datasets, args.models, args.sentence_level, args.deep_finetuning, args.n_epochs, args.learning_rate, args.keep_model


if __name__ == '__main__':
    device, seeds, tag_schemes, splits, datasets, models, sentence_level, deep_finetuning, n_epochs, lr, keep_model = parse_args()
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
    from models.bert_model import BertCRFNERModel
    from models.bilstm_model import BiLSTMNERModel
    from utils.data import NERData
    
    torch.device('cuda' if gpu else 'cpu')
    torch.backends.cudnn.deterministic = True

    seeds = [int(seed) for seed in seeds.split(',')]
    tag_schemes = [str(tag_scheme) for tag_scheme in tag_schemes.split(',')]
    splits = [int(split) for split in splits.split(',')]
    datasets = [str(dataset) for dataset in datasets.split(',')]
    models = [str(model) for model in models.split(',')]

    datafiles = {'solid_state': 'data/solid_state.json',
                 'doping': 'data/doping.json',
                 'aunp2': 'data/aunp_2lab.json',
                 'aunp11': 'data/aunp_11lab.json'}
    modelfiles = {'bert': 'bert-base-uncased',
                  'scibert': 'allenai/scibert_scivocab_uncased',
                  'matbert': '/home/amalie/MatBERT_NER/matbert_ner/matbert-base-uncased'}

    for seed in seeds:
        for tag_scheme in tag_schemes:
            for split in splits:
                for dataset in datasets:
                    for model in models:
                        alias = '{}_{}_{}_{}_crf_{}_{}_{}'.format(model, dataset, 'sentence' if sentence_level else 'paragraph', 'deep' if deep_finetuning else 'shallow', tag_scheme.lower(), seed, split)
                        save_dir = os.getcwd()+'/{}/'.format(alias)
                        print('calculating results for {}'.format(alias))
                        # try:
                        if os.path.exists(save_dir+'test.pt'):
                            print('already calculated {}, skipping'.format(alias))
                        else:
                            if not os.path.exists(save_dir):
                                os.mkdir(save_dir)
                        ner_data = NERData(modelfiles[model], tag_format=tag_format)
                        ner_data.preprocess(datafiles[dataset], (0.1, split/800, split/100), is_file=True, sentence_level=sentence_level, shuffle=True, seed=seed)
                        ner_data.create_dataloaders(batch_size=32)
                        classes = ner_data.classes
                        torch.save(classes, save_dir+'classes.pt')
                        ner_model = BertCRFNERModel(modelname=modelfiles[model], classes=classes, tag_format=tag_format, device=device, lr=lr)
                        ner_model.train(n_epochs, ner_data.dataloaders['train'], val_dataloader=ner_data.dataloaders['valid'], dev_dataloader=ner_data.dataloaders['test'],
                                        save_dir=save_dir, deep_finetuning=deep_finetuning)
                        epoch_files = glob.glob(save_dir+'epoch_*pt')
                        for f in epoch_files:
                            try:
                                os.remove(f)
                            except:
                                print('error while deleting file: {}'.format(f))
                        if not keep_model:
                            try:
                                os.remove(save_dir+'best.pt')
                            except:
                                print('error while deleting file: {}best.pt'.format(savedir))
                        # except:
                        #     print('error calculating results for {}'.format(alias))                
