from .models.base_ner_model import NERModel
import json

model = NERModel(
    model="../../matbert-base-uncased"
)

model.train('./matbert_ner/data/ner_annotations.json')
