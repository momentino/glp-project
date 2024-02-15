import json
import pandas as pd

dataset_path="./datasets/combined_aro_valse.json"
correct_subset_path='./datasets/correct_subset.json'
wrong_subset_path='./datasets/wrong_subset.json'
statistics_csv_path='./datasets/subset_statistics.csv'

def subset_statistics(subset, dataset_path, subset_path, csv_path):

    actual_aro=0
    actual_valse=0
    pred_aro=0
    pred_valse=0
    with open(dataset_path, 'r') as f:
        full_data = json.load(f)
    
    if subset=='correct':
        for key,value in full_data.items():
            if value['surprisal_difference']<2 and value["dataset"]=="ARO":
                pred_aro+=1
            elif value['surprisal_difference']<2 and value["dataset"]=="VALSE":
                pred_valse+=1

    else:
        for key,value in full_data.items():
            if value['surprisal_difference']>5 and value["dataset"]=="ARO":
                pred_aro+=1
            elif value['surprisal_difference']>5 and value["dataset"]=="VALSE":
                pred_valse+=1

    with open(subset_path, 'r') as f:
        subset_data = json.load(f)

    size=len(subset_data)
    
    for key,value in subset_data.items():
        if value["dataset"]=="ARO":
            actual_aro+=1
        else:
            actual_valse+=1

    aro_prop=round(100*actual_aro/size,4)
    valse_prop=round(100*actual_valse/size,4)
    predicted=pred_aro+pred_valse
    aro_acc=round(100*actual_aro/pred_aro,4)
    valse_acc=round(100*actual_valse/pred_valse,4)

    df = pd.read_csv(csv_path)
    new_row = [{
        'subset': subset, 
        'actual size': size,
        'ARO proportion': aro_prop,
        'VALSE proportion': valse_prop,
        'predicted': predicted,
        'ARO acc': aro_acc,
        'VALSE acc': valse_acc
    }]
   
    row = pd.DataFrame(new_row)
    df = pd.concat([df, row], ignore_index=True)
    df.to_csv(csv_path, index=False)

if __name__ == '__main__':
    subset_statistics('correct',dataset_path,correct_subset_path,statistics_csv_path)
    subset_statistics('wrong',dataset_path,wrong_subset_path,statistics_csv_path)