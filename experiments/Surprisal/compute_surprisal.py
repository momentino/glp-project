from surprisal_utils import *
from transformers import RobertaForMaskedLM, RobertaTokenizer
import numpy as np
import json

model_name = 'roberta-base'
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

dataset_path="./datasets/combined_aro_valse.json"
correct_subset_path='./datasets/correct_subset.json'
wrong_subset_path='./datasets/wrong_subset.json'

correct_subset={}
wrong_subset={}


with open(dataset_path, 'r') as f:
    data = json.load(f)

for key,value in data.items():
    if value['surprisal_difference']<2:
        correct_subset[key]=value
        
    elif value['surprisal_difference']>5:
        wrong_subset[key]=value
    
print(len(correct_subset))
print(len(wrong_subset))

        
with open(correct_subset_path,'w') as f:
    json.dump(correct_subset, f, indent=4)

with open(wrong_subset_path,'w') as f:
    json.dump(wrong_subset, f, indent=4)