# MatBERT NER

A framework for materials science NER using the HuggingFace Transformers NLP Toolkit.

# Installation

```git
git clone https://github.com/walkernr/MatBERT_NER.git MatBERT_NER
cd MatBERT_NER
pip install -r requirements.txt .
```

# Example Usage

The folowing command will train the MatBERT model on the solid state dataset using default parameters

```
python train.py -dv gpu:0 -ds solid_state -ml matbert
```

Additional parameters can be specified.

```
usage: train.py [-h] [-dv DEVICE] [-sd SEEDS] [-ts TAG_SCHEMES] [-st SPLITS] [-ds DATASETS] [-ml MODELS] [-sl] [-bs BATCH_SIZE] [-on OPTIMIZER_NAME] [-wd WEIGHT_DECAY] [-ne N_EPOCH]
                [-eu EMBEDDING_UNFREEZE] [-tu TRANSFORMER_UNFREEZE] [-el EMBEDDING_LEARNING_RATE] [-tl TRANSFORMER_LEARNING_RATE] [-cl CLASSIFIER_LEARNING_RATE] [-sf SCHEDULING_FUNCTION]   
                [-km]

optional arguments:
  -h, --help            show this help message and exit
  -dv DEVICE, --device DEVICE
                        computation device for model (e.g. cpu, gpu:0, gpu:1)
  -sd SEEDS, --seeds SEEDS
                        comma-separated seeds for data shuffling and model initialization (e.g. 1,2,3 or 2,4,8)
  -ts TAG_SCHEMES, --tag_schemes TAG_SCHEMES
                        comma-separated tagging schemes to be considered (e.g. iob1,iob2,iobes)
  -st SPLITS, --splits SPLITS
                        comma-separated training splits to be considered, in percent (e.g. 80). test split will always be 10% and the validation split will be 1/8 of the training split   
                        unless the training split is 100%
  -ds DATASETS, --datasets DATASETS
                        comma-separated datasets to be considered (e.g. solid_state,doping)
  -ml MODELS, --models MODELS
                        comma-separated models to be considered (e.g. matbert,scibert,bert)
  -sl, --sentence_level
                        switch for sentence-level learning instead of paragraph-level
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        number of samples in each batch
  -on OPTIMIZER_NAME, --optimizer_name OPTIMIZER_NAME
                        name of optimizer, add "_lookahead" to implement lookahead on top of optimizer (not recommended for ranger or rangerlars)
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay for optimizer (excluding bias, gamma, and beta)
  -ne N_EPOCH, --n_epoch N_EPOCH
                        number of training epochs
  -eu EMBEDDING_UNFREEZE, --embedding_unfreeze EMBEDDING_UNFREEZE
                        epoch (index) at which bert embeddings are unfrozen
  -tu TRANSFORMER_UNFREEZE, --transformer_unfreeze TRANSFORMER_UNFREEZE
                        comma-separated number of transformers (encoders) to unfreeze at each epoch
  -el EMBEDDING_LEARNING_RATE, --embedding_learning_rate EMBEDDING_LEARNING_RATE
                        embedding learning rate
  -tl TRANSFORMER_LEARNING_RATE, --transformer_learning_rate TRANSFORMER_LEARNING_RATE
                        transformer learning rate
  -cl CLASSIFIER_LEARNING_RATE, --classifier_learning_rate CLASSIFIER_LEARNING_RATE
                        pooler/classifier learning rate
  -sf SCHEDULING_FUNCTION, --scheduling_function SCHEDULING_FUNCTION
                        function for learning rate scheduler (linear, exponential, or cosine)
  -km, --keep_model     switch for saving the best model parameters to disk
```

To train on custom annotated datasets, the `train.py` script has a dictionary `data_files` where additional datasets can be specified. Similarly, alternative pre-trained models can be used by modifying the `model_files` dictionary.

For prediction, the `predict` function contained within `predict.py` can be used. An example that was used internally can be found in the `predict_script.py` file. Furthermore, an example utilizing MongoDB can be found in the `predict_mongo.py` script. Note that these two examples will need to be edited for your specific needs to be usable.

# License
