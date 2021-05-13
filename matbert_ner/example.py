import os
import argparse
import glob
import numpy as np
from seqeval.scheme import IOB1, IOB2, IOBES
from seqeval.metrics import classification_report


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dv', '--device', help='computation device for model (e.g. cpu, gpu:0, gpu:1)', type=str, default='gpu:0')
    parser.add_argument('-sd', '--seeds', help='comma-separated seeds for data shuffling and model initialization (e.g. 1,2,3 or 2,4,8)', type=str, default='256')
    parser.add_argument('-ts', '--tag_schemes', help='comma-separated tagging schemes to be considered (e.g. IOB1,IOB2,IOBES)', type=str, default='IOBES')
    parser.add_argument('-st', '--splits', help='comma-separated training splits to be considered, in percent (e.g. 80). test split will always be 10%% and the validation split will be 1/8 of the training split', type=str, default='80')
    parser.add_argument('-ds', '--datasets', help='comma-separated datasets to be considered (e.g. solid_state,doping)', type=str, default='solid_state')
    parser.add_argument('-ml', '--models', help='comma-separated models to be considered (e.g. matbert,scibert,bert)', type=str, default='matbert')
    parser.add_argument('-sl', '--sentence_level', help='switch for sentence-level learning instead of paragraph-level', action='store_true')
    parser.add_argument('-bs', '--batch_size', help='number of samples in each batch', type=int, default=32)
    parser.add_argument('-ne', '--n_epochs', help='number of training epochs', type=int, default=16)
    parser.add_argument('-fe', '--frozen_epochs', help='number of frozen bert epochs', type=int, default=0)
    parser.add_argument('-tl', '--transformer_learning_rate', help='transformer learning rate', type=float, default=2e-4)
    parser.add_argument('-cl', '--classifier_learning_rate', help='classifier learning rate', type=float, default=1e-3)
    parser.add_argument('-km', '--keep_model', help='switch for saving the best model parameters to disk', action='store_true')
    args = parser.parse_args()
    return args.device, args.seeds, args.tag_schemes, args.splits, args.datasets, args.models, args.sentence_level, args.batch_size, args.n_epochs, args.frozen_epochs, args.transformer_learning_rate, args.classifier_learning_rate, args.keep_model


if __name__ == '__main__':
    device, seeds, tag_schemes, splits, datasets, models, sentence_level, batch_size, n_epochs, frozen_epochs, tlr, clr, keep_model = parse_args()
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
    schemes = {'IOB1': IOB1, 'IOB2': IOB2, 'IOBES': IOBES}

    for seed in seeds:
        for tag_scheme in tag_schemes:
            for split in splits:
                for dataset in datasets:
                    for model in models:
                        alias = '{}_{}_{}_{}_crf_{}_{}_{}_{:.0e}_{:.0e}_{}_{}'.format(model, dataset, 'sentence' if sentence_level else 'paragraph', tag_scheme.lower(), batch_size, n_epochs, frozen_epochs, tlr, clr, seed, split)
                        save_dir = os.getcwd()+'/{}/'.format(alias)
                        print('calculating results for {}'.format(alias))
                        # try:
                        if os.path.exists(save_dir+'test.pt'):
                            print('already calculated {}, skipping'.format(alias))
                        else:
                            if not os.path.exists(save_dir):
                                os.mkdir(save_dir)
                            
                            torch.manual_seed(seed)
                            torch.cuda.manual_seed(seed)
                            ner_data = NERData(modelfiles[model], tag_scheme=tag_scheme)
                            ner_data.preprocess(datafiles[dataset], (0.1, split/800, split/100), is_file=True, sentence_level=sentence_level, shuffle=True, seed=seed)
                            ner_data.create_dataloaders(batch_size=batch_size)
                            classes = ner_data.classes
                            torch.save(classes, save_dir+'classes.pt')

                            torch.manual_seed(seed)
                            torch.cuda.manual_seed(seed)
                            ner_model = BertCRFNERModel(modelname=modelfiles[model], classes=classes, tag_scheme=tag_scheme, device=device, tlr=tlr, clr=clr)
                            print(ner_model.model)
                            ner_model.train(n_epochs, ner_data.dataloaders['train'], val_dataloader=ner_data.dataloaders['valid'], dev_dataloader=ner_data.dataloaders['test'],
                                            save_dir=save_dir, frozen_transformer_epochs=frozen_epochs)
                            
                            _, _, _, _, labels, predictions = torch.load(save_dir+'test.pt')
                            print(classification_report(labels, predictions, mode='strict', scheme=schemes[tag_scheme]))

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
