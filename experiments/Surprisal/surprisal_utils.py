import torch
import torch.nn.functional as F
import json

def sent_surprisal(logits, token_ids):
    size = token_ids.shape[0]
    #compute probabilities
    probabilities = F.softmax(logits, dim=-1)
    # Compute the surprisal
    surprisal = -torch.log(probabilities[torch.arange(size),:, token_ids[0]])
    return surprisal.mean() # Average the surprisal values

def load_foils(json_path):
    foils = []
    with open(json_path, 'r') as f:
        data = json.load(f)
    foils=[sample['foil_active'] for sample in data.values()]
    return foils

def compute_surprisal(sentences,tokenizer,model):
    # Initialize the list to store the surprisal values
    surprisal_scores = []
    # Iterate through the dataset
    for sent in sentences:
        # Tokenize the input
        input_ids = torch.tensor(tokenizer.encode(sent,add_special_tokens=True)).unsqueeze(0)
        # Forward pass to compute the layer outputs
        with torch.no_grad():
            outputs = model(input_ids)
            logits=outputs.logits
            # Compute the surprisal
        surprisal = sent_surprisal(logits, input_ids)
        surprisal_scores.append(surprisal)

    return surprisal_scores