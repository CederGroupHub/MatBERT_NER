from models.bert_model import BertCRFNERModel
from pprint import pprint

""" Sequence of events:
    1. input sentence(s), labels, and trained_model
    2. tokenize sentence(s) and prepare in dataset
    3. preprocess """
sentence = """The gold nanorods were synthesized using the typical citrate reduction method. First, spherical Au 
    nanoparticle seeds were added to a beaker."""
sentences = ["""The morphology of the gold nanorods was investigated using TEM. Figure X reveals the dumbbell-like
             nanorods had an aspect ratio of about 5""",
             """Cubic gold nanoparticles were synthesized according to the literature. Precursors were purchased from 
             Sigma\xa0Aldrich, and the spherical Au seeds were synthesized the previous night."""]

labels = ['MOR', 'DES']

model = BertCRFNERModel(
    modelname="../../Scrap/matbert-base-uncased"
)

preds = model.predict(sentence, labels=['MOR', 'DES'], trained_model="../../Scrap/best.pt")
pprint(preds)
