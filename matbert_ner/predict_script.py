from predict import predict
import json

doping_data = '../../datasets/dop_toparse_169828.json'
aunp_data = '../../datasets/aunp_recipes_characterization_filtered.json'
example_data = '../../../LBL_NER_DATASETS/example.json'
model = '../../matbert-base-uncased'
solid_state_state = '../../MatBERT_NER_models/matbert_solid_state_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_100/best.pt'
doping_state = '../../MatBERT_NER_models/matbert_doping_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_100/best.pt'
aunp6_state = '../../MatBERT_NER_models/matbert_aunp6_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_100/best.pt'

# predict(doping_data, True, model, solid_state_state, predict_path=solid_state_state.replace('best.pt', 'predict_doping_solid_state_169828.pt'), device='gpu:0')
# predict(doping_data, True, model, doping_state, predict_path=doping_state.replace('best.pt', 'predict_doping_doping_169828.pt'), device='gpu:0')
# predict(aunp_data, True, model, solid_state_state, predict_path=solid_state_state.replace('best.pt', 'predict_aunp_solid_state.pt'), device='gpu:0')
# predict(aunp_data, True, model, aunp6_state, predict_path=aunp6_state.replace('best.pt', 'predict_aunp_aunp6.pt'), device='gpu:0')
predict_path = solid_state_state.replace('best.pt', 'predict_example.json')
example_data = json.load(open(example_data, 'r'))
x = predict(example_data, False, model, solid_state_state, predict_path=predict_path, device='cpu')
print(json.dumps(x, indent=2))