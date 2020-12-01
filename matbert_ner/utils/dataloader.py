import json
import torch
from torch.utils.data import TensorDataset
from utils.data import InputExample, convert_examples_to_features

def load_data(datafile,tokenizer):
	data = []
	with open(datafile, 'r') as f:
		for l in f:
			data.append(json.loads(l))

	classes_raw = data[0]['labels']
	classes = ["O"]
	for c in classes_raw:
		classes.append("B-{}".format(c))
		classes.append("I-{}".format(c))

	print(classes)
	data = [[(d['text'],d['annotation']) for d in s] for a in data for s in a['tokens']]

	input_examples = []
	max_sequence_length = 0
	for i, d in enumerate(data):
		labels = []
		text = []
		for t,l in d:

			#This causes issues with BERT for some reason
			if t in ['̄','̊']:
				continue	

			text.append(t)
			if l is None:
				label = "O"
			elif "PUT" in l or "PVL" in l:
				label = "O"
			else:
				if len(labels) > 0 and l in labels[-1]:
					label = "I-{}".format(l)
				else:
					label = "B-{}".format(l)
			labels.append(label)

		if len(text) > max_sequence_length:
			max_sequence_length = len(text)

		example = InputExample(i, text, labels)

		input_examples.append(example)

	features = convert_examples_to_features(
	        input_examples,
	        classes,
	        max_sequence_length,
	        tokenizer,
	)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_valid_mask = torch.tensor([f.valid_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

	dataset = TensorDataset(all_input_ids, all_input_mask, all_valid_mask, all_segment_ids, all_label_ids)

	return dataset
