from predict import predict

data = '../../datasets/dop_toparse_169828.json'
model = '../../matbert-base-uncased'
solid_state_state = './matbert_solid_state_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_100/best.pt'
doping_state = './matbert_doping_paragraph_iobes_crf_10_lamb_5_1_012_1e-04_2e-03_1e-02_0e+00_exponential_256_100/best.pt'

predict(data, True, model, solid_state_state, predict_path='{}/{}'.format(solid_state_state, 'predict_doping_solid_state_169828.pt'))
predict(data, True, model, doping_state, predict_path='{}/{}'.format(doping_state, 'predict_doping_doping_169828.pt'))
