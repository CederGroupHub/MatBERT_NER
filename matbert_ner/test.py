from .models.base_ner_model import NERModel

model = NERModel(
    model="./matbert_ner/models/matbert-base-uncased",
    classes=['O', 'B-MOR', 'I-MOR', 'B-DES', 'I-DES'],
    trained_ner='./matbert_ner/models/matbert-uncased_5e-05_10_best.pt'
)
sentence = "The gold nanorods were synthesized using the typical citrate reduction method. First, spherical Au nanoparticle seeds were introduced."
preds = model.predict(sentence)
print(preds)
