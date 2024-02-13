from surprisal_utils import *
from transformers import RobertaForMaskedLM, RobertaTokenizer
import numpy as np

model_name = 'roberta-base'
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

threshold_path="./datasets/threshold.json"
dataset_path="./datasets/combined_aro_valse.json"

threshold_foils=load_foils(threshold_path)
threshold_surprisals=compute_surprisal(threshold_foils,tokenizer,model)

sents=["A woman bothers the usher.","An usher bothers the woman.","The food packages a man.","I eat pizza.","A pen eats a table."]
surprisals=compute_surprisal(sents,tokenizer,model)
print(surprisals)
#min_surprisal=(min(threshold_surprisals))
#max_surprisal=(max(threshold_surprisals))
#max_index=np.argmax(threshold_surprisals)
#print(min_surprisal)
#print(max_surprisal)
#print(max_index)