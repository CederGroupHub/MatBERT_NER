from models.base_ner_model import NERModel
import json

model = NERModel(
    model="./matbert_ner/models/matbert-base-uncased"
)

model.train('./matbert_ner/data/impurityphase_fullparas.json')
