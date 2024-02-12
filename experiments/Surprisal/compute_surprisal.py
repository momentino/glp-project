from surprisal_utils import *
from transformers import RobertaForMaskedLM, RobertaTokenizer, BertForMaskedLM, BertTokenizer

model_name = 'roberta-base'
model = RobertaForMaskedLM.from_pretrained(model_name)
tokenizer = RobertaTokenizer.from_pretrained(model_name)

threshold_path="./datasets/threshold.json"
dataset_path="./datasets/combined_aro_valse.json"

#threshold_foils=load_foils(threshold_path)
#threshold_surprisals=compute_surprisal(threshold_foils,tokenizer,model)

sents=["Colorless green ideas sleep furiously"]
surprisals=compute_surprisal(sents,tokenizer,model)
print(surprisals)
#min_surprisal=(min(threshold_surprisals))
#max_surprisal=(max(threshold_surprisals))
#print(min_surprisal)
#print(max_surprisal)