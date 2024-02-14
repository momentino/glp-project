from surprisal_utils import *
from transformers import RobertaForMaskedLM, RobertaTokenizer
import numpy as np
import json

model_name = 'roberta-base'
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

threshold_path="./datasets/threshold.json"
dataset_path="./datasets/combined_aro_valse.json"

with open(dataset_path, 'r') as f:
    data = json.load(f)

for sample in data.values():
    foil=sample['foil_active']
    surprisal=sentence_surprisal(foil,tokenizer,model)
    sample['surprisal']=surprisal

with open(dataset_path, 'w') as f:
    json.dump(data, f, indent=4)