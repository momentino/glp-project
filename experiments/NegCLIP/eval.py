from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

def eval(model, loader):
    model.model.eval()

    c_scores = []
    f_scores = []
    scores_by_cat = dict()
    total_num_samples = 0
    with torch.no_grad():
        for images, captions, foils, categories in tqdm(loader):
            for image, caption, foil, category in zip(images,captions,foils,categories):
                image_features = model.model.encode_image(image.unsqueeze(0))
                text_input = torch.stack([caption.squeeze(0),foil.squeeze(0)], dim=0)
                text_features = model.model.encode_text(text_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

                caption_score = similarity[0][0]
                foil_score = similarity[0][1]
                c_scores.extend([caption_score])
                f_scores.extend([foil_score])
                # this is to iterate multiple lists together
                try:
                    scores_by_cat[category]['caption_scores'].append(caption_score)
                    scores_by_cat[category]['foil_scores'].append(foil_score)
                except:
                    scores_by_cat[category] = {
                        'caption_scores': [caption_score],
                        'foil_scores': [foil_score]
                    }
                total_num_samples += 1

    pairwise_acc = sum(
        [1 if c_scores[i].item() > f_scores[i].item() else 0 for i in
         range(len(c_scores))]) / total_num_samples
    pairwise_acc_50 = sum(
        [1 if c_scores[i].item() > f_scores[i].item() and c_scores[i].item() > 0.5 else 0 for i in
         range(len(c_scores))]) / total_num_samples
    pairwise_acc_60 = sum(
        [1 if c_scores[i].item() > f_scores[i].item() and c_scores[i].item() > 0.6 else 0 for i in
         range(len(c_scores))]) / total_num_samples
    pairwise_acc_70 = sum(
        [1 if c_scores[i].item() > f_scores[i].item() and c_scores[i].item() > 0.7 else 0 for i in
         range(len(c_scores))]) / total_num_samples
    precision_caption = -1
    precision_foil =-1
    acc = -1

    # Now performance aggregated by verb category
    perf_by_category = {}
    for key, value in scores_by_cat.items():
        cat_c_scores = value['caption_scores']
        cat_f_scores = value['foil_scores']
        num_samples_by_cat = len(cat_f_scores)
        try:
            perf_by_category[key]['pairwise_acc'] = sum(
                [1 if cat_c_scores[i].item() > cat_f_scores[i].item() else 0 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['pairwise_acc_50'] = sum(
                [1 if cat_c_scores[i].item() > cat_f_scores[i].item() and cat_c_scores[i].item() > 0.5 else 0
                 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['pairwise_acc_60'] = sum(
                [1 if cat_c_scores[i][1].item() > cat_f_scores[i][1].item() and cat_c_scores[i].item() > 0.6 else 0
                 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['pairwise_acc_70'] = sum(
                [1 if cat_c_scores[i][1].item() > cat_f_scores[i][1].item() and cat_c_scores[i].item() > 0.7 else 0
                 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['precision_caption'] = -1
            perf_by_category[key]['precision_foil'] = -1

            perf_by_category[key]['acc'] = -1
        except:
            perf_by_category[key] = {'pairwise_acc': float()}
            perf_by_category[key]['pairwise_acc'] = sum(
                [1 if cat_c_scores[i].item() > cat_f_scores[i].item() else 0 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['pairwise_acc_50'] = sum(
                [1 if cat_c_scores[i].item() > cat_f_scores[i].item() and cat_c_scores[i].item() > 0.5 else 0
                 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['pairwise_acc_60'] = sum(
                [1 if cat_c_scores[i].item() > cat_f_scores[i].item() and cat_c_scores[i].item() > 0.6 else 0
                 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['pairwise_acc_70'] = sum(
                [1 if cat_c_scores[i].item() > cat_f_scores[i].item() and cat_c_scores[i].item() > 0.7 else 0
                 for i in
                 range(len(cat_c_scores))]) / num_samples_by_cat
            perf_by_category[key]['precision_caption'] = -1
            perf_by_category[key]['precision_foil'] = -1
            perf_by_category[key]['acc'] = -1

    return (acc,
            pairwise_acc,
            pairwise_acc_50,
            pairwise_acc_60,
            pairwise_acc_70,
            precision_caption,
            precision_foil,
            perf_by_category)