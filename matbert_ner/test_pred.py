from models.bert_model import BertCRFNERModel
from utils.data import NERData
from torch.utils.data import DataLoader
from pprint import pprint


sentence = """The gold nanorods were synthesized using the typical citrate reduction method. First, spherical Au 
    nanoparticle seeds were added to a beaker."""
sentences = ["""The morphology of the gold nanorods was investigated using TEM. Figure X reveals the dumbbell-like
             nanorods had an aspect ratio of about 5""",
             """Cubic gold nanoparticles were synthesized according to the literature. Precursors were purchased from 
             Sigma\xa0Aldrich, and the spherical Au seeds were synthesized the previous night."""]

modelname = "../../Scrap/matbert-base-uncased"
trained_model = "../../Scrap/best.pt"

labels = ['MOR', 'DES']

model = BertCRFNERModel(
    modelname=modelname
)

# test single sentence
preds = model.predict(sentence, labels=['MOR', 'DES'], trained_model=trained_model)
print("Single sentence predictions: \n")
pprint(preds)
print("\n")

# test list of sentences
preds = model.predict(sentences, labels=['MOR', 'DES'], trained_model=trained_model)
print("Single sentence predictions: \n")
pprint(preds)
print("\n")

# test preprocessed dataloader
ner_data = NERData(modelname)

tokenized_dataset = []
for para in sentences:
    token_set = ner_data.create_tokenset(para)
    token_set['labels'] = labels
    tokenized_dataset.append(token_set)

ner_data.preprocess(tokenized_dataset, is_file=False)
tensor_dataset = ner_data.dataset
pred_dataloader = DataLoader(tensor_dataset)

model = BertCRFNERModel(
    modelname=modelname,
    classes = ner_data.classes
)
preds = model.predict(pred_dataloader, trained_model=trained_model, tok_dataset=tokenized_dataset)
print("Preprocessed dataloader predictions: \n")
pprint(preds)
print("\n")
