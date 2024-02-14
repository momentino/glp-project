from tqdm import tqdm
from models.AdaptedModels.ALBEFForSimilarities import ALBEFForSimilarities
from utils.utils import load_weights

import torch
import torch.nn.functional as F
import numpy as np


def similarities(model, loader, config):
    adapted_model = ALBEFForSimilarities(model)
    load_weights(adapted_model.base_model, model_name='ALBEF', general_config=config) # we load the weights of the base architecture
    adapted_model.eval()

    tf_scores=[]
    ap_scores=[]
    diff_scores=[] #cos(true_act,foil_act) - cos(true_act,true_pass). Ideally we want it to be negative (true act/pass more similar than true/foil)

    scores_by_cat = dict()

    with torch.no_grad():
        for images, true_actives, foil_actives, true_passives, categories in tqdm(loader):
            true_actives_vl_embeds, foil_actives_vl_embeds, true_passives_vl_embeds = adapted_model(images, true_actives, foil_actives, true_passives)
            
            tf_similarities=F.cosine_similarity(true_actives_vl_embeds,foil_actives_vl_embeds,dim=1)
            ap_similarities=F.cosine_similarity(true_actives_vl_embeds,true_passives_vl_embeds,dim=1)
            tf_scores.extend(tf_similarities)
            ap_scores.extend(ap_similarities)
            diff_scores.extend(tf_similarities-ap_similarities)

            # this is to iterate multiple lists together
            for cat, tf_sc,ap_sc,d_sc in zip(categories, tf_scores, ap_scores, diff_scores):
                try:
                    scores_by_cat[cat]['true_foil_scores'].append(tf_sc)
                    scores_by_cat[cat]['active_passive_scores'].append(ap_sc)
                    scores_by_cat[cat]['difference_scores'].append(d_sc)
                except:
                    scores_by_cat[cat] = {
                        'true_foil_scores': [tf_sc],
                        'active_passive_scores': [ap_sc],
                        'difference_scores':[d_sc]
                    }

    tf_mean=np.mean(tf_scores)
    tf_std=np.std(tf_scores)

    ap_mean=np.mean(ap_scores)
    ap_std=np.std(ap_scores)

    diff_mean=np.mean(diff_scores)
    diff_std=np.std(diff_scores)
        
    # Now performance aggregated by verb category
    perf_by_category = {}
    for key, value in scores_by_cat.items():
        cat_tf_scores = value['true_foil_scores']
        cat_ap_scores = value['active_passive_scores']
        cat_diff_scores=value['difference_scores']
    
        try:
            perf_by_category[key]['true_foil_mean'] = np.mean(cat_tf_scores)
            perf_by_category[key]['true_foil_std'] = np.std(cat_tf_scores)
            perf_by_category[key]['active_passive_mean'] = np.mean(cat_ap_scores)
            perf_by_category[key]['active_passive_std'] = np.std(cat_ap_scores)
            perf_by_category[key]['difference_mean'] = np.mean(cat_diff_scores)
            perf_by_category[key]['difference_std'] = np.std(cat_diff_scores)
        except:
            perf_by_category[key] = {
                'true_foil_mean' : np.mean(cat_tf_scores),
                'true_foil_std' : np.std(cat_tf_scores), 
                'active_passive_mean' : np.mean(cat_ap_scores),
                'active_passive_std' : np.std(cat_ap_scores),
                'difference_mean' : np.mean(cat_diff_scores),
                'difference_std' : np.std(cat_diff_scores)
            }

    return (tf_mean,
            tf_std,
            ap_mean,
            ap_std,
            diff_mean,
            diff_std,
            perf_by_category)