from tqdm import tqdm
from models.AdaptedModels.ALBEFForITM import ALBEFForITM
from utils.utils import load_weights

import torch


def eval(model, loader, config):
    adapted_model = ALBEFForITM(model)
    load_weights(adapted_model.base_model, model_name='ALBEF', general_config=config) # we load the weights of the base architecture
    adapted_model.eval()

    c_scores = []
    f_scores = []
    total_num_samples = 0
    with torch.no_grad():
        for images, captions, foils, good_caption in tqdm(loader):
            if good_caption < 2:
                continue
            caption_scores, foils_scores = adapted_model(images, captions, foils)
            c_scores.extend(caption_scores)
            f_scores.extend(foils_scores)
            total_num_samples+=1


    pairwise_acc = sum([1 if c_scores[i][1].item()>f_scores[i][1].item() else 0 for i in range(len(c_scores))])/total_num_samples
    pairwise_acc_50 = sum([1 if c_scores[i][1].item()>f_scores[i][1].item() and c_scores[i][1].item()>0.5 else 0 for i in range(len(c_scores))])/total_num_samples
    pairwise_acc_60 = sum([1 if c_scores[i][1].item() > f_scores[i][1].item() and c_scores[i][1].item() > 0.6 else 0 for i in
                           range(len(c_scores))]) / total_num_samples
    pairwise_acc_70 = sum([1 if c_scores[i][1].item() > f_scores[i][1].item() and c_scores[i][1].item() > 0.7 else 0 for i in
                           range(len(c_scores))]) / total_num_samples
    precision_caption = sum([1 if c_scores[i][0].item()<c_scores[i][1].item() else 0 for i in range(len(c_scores))])/total_num_samples # "the caption fits the image well (VALSE)"
    precision_foil = sum([1 if f_scores[i][0].item()>=f_scores[i][1].item() else 0 for i in range(len(f_scores))])/total_num_samples # the foil doesn't fit the image well
    acc = sum([(1 if (c_scores[i][0].item()<c_scores[i][1].item() ) else 0) + (1 if (f_scores[i][0].item()>=f_scores[i][1].item()) else 0) for i in range(len(c_scores))])/(total_num_samples*2)



    return (acc,
            pairwise_acc,
            pairwise_acc_50,
            pairwise_acc_60,
            pairwise_acc_70,
            precision_caption,
            precision_foil
            )

