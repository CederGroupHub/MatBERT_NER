from .models.base_ner_model import NERModel

model = NERModel(
    model="../../matbert-base-uncased",
    labels=['MOR', 'DES'],
    trained_ner='../../matbert-uncased_5e-05_10_best.pt'
)
sentence = "The gold nanorods were synthesized using the typical citrate reduction method. First, spherical Au nanoparticle seeds were introduced."
preds = model.predict(sentence)
print(preds)
