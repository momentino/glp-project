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

threshold_1=set()
threshold_2=set()

i=0
for sample in data.values():
    if sample['surprisal_difference']>4:
        threshold_1.add(sample['foil_active'])
    if sample['surprisal_difference']>3:
        threshold_2.add(sample['foil_active'])
        i+=1
        
diff_2_1 = threshold_2.difference(threshold_1)

for item in diff_2_1:
    print(item)

print(i)