from tqdm import tqdm
from models.AdaptedModels.BLIPForSimilarities import BLIPForSimilarities
from utils.utils import load_weights

import torch
import torch.nn.functional as F
import numpy as np

def similarities(model, loader, config):
    adapted_model = BLIPForSimilarities(model)
    load_weights(adapted_model.base_model, model_name='BLIP', general_config=config) # we load the weights of the base architecture
    adapted_model.eval()

    tf_vl_scores=[]
    ap_vl_scores=[]
    diff_vl_scores=[] #cos(true_act,foil_act) - cos(true_act,true_pass). Ideally we want it to be negative (true act/pass more similar than true/foil)

    scores_by_cat = dict()

    with torch.no_grad():
        for images, true_actives, foil_actives, true_passives, categories in tqdm(loader):
            true_actives_vl_embeds, foil_actives_vl_embeds, true_passives_vl_embeds = adapted_model(images, true_actives, foil_actives, true_passives)
        
            tf_vl_similarities=F.cosine_similarity(true_actives_vl_embeds,foil_actives_vl_embeds,dim=-1)
            ap_vl_similarities=F.cosine_similarity(true_actives_vl_embeds,true_passives_vl_embeds,dim=-1)

            tf_vl_scores.extend(tf_vl_similarities)
            ap_vl_scores.extend(ap_vl_similarities)
            diff_vl_scores.extend(tf_vl_similarities-ap_vl_similarities)

            # this is to iterate multiple lists together
            for cat, tf_vl_sc,ap_vl_sc,d_vl_sc in zip(categories, tf_vl_scores, ap_vl_scores, diff_vl_scores):
                try:
                    scores_by_cat[cat]['true_foil_vl_scores'].append(tf_vl_sc)
                    scores_by_cat[cat]['active_passive_vl_scores'].append(ap_vl_sc)
                    scores_by_cat[cat]['difference_vl_scores'].append(d_vl_sc)
                except:
                    scores_by_cat[cat] = {
                        'true_foil_vl_scores': [tf_vl_sc],
                        'active_passive_vl_scores': [ap_vl_sc],
                        'difference_vl_scores':[d_vl_sc]
                    }

    tf_text_mean=None #so that they are returned and we don't have to make distinct cases in the experiment file
    tf_text_std=None
    ap_text_mean=None
    ap_text_std=None
    diff_text_mean=None
    diff_text_std=None

    tf_vl_mean=np.mean(tf_vl_scores)
    tf_vl_std=np.std(tf_vl_scores)
    ap_vl_mean=np.mean(ap_vl_scores)
    ap_vl_std=np.std(ap_vl_scores)
    diff_vl_mean=np.mean(diff_vl_scores)
    diff_vl_std=np.std(diff_vl_scores)
        
    # Now performance aggregated by verb category
    perf_by_category = {}
    for key, value in scores_by_cat.items():
        cat_tf_vl_scores = value['true_foil_vl_scores']
        cat_ap_vl_scores = value['active_passive_vl_scores']
        cat_diff_vl_scores = value['difference_vl_scores']
    
        try:
            perf_by_category[key]['true_foil_text_mean'] = None
            perf_by_category[key]['true_foil_text_std'] = None
            perf_by_category[key]['active_passive_text_mean'] = None
            perf_by_category[key]['active_passive_text_std'] = None
            perf_by_category[key]['difference_text_mean'] = None
            perf_by_category[key]['difference_text_std'] = None

            perf_by_category[key]['true_foil_vl_mean'] = np.mean(cat_tf_vl_scores)
            perf_by_category[key]['true_foil_vl_std'] = np.std(cat_tf_vl_scores)
            perf_by_category[key]['active_passive_vl_mean'] = np.mean(cat_ap_vl_scores)
            perf_by_category[key]['active_passive_vl_std'] = np.std(cat_ap_vl_scores)
            perf_by_category[key]['difference_vl_mean'] = np.mean(cat_diff_vl_scores)
            perf_by_category[key]['difference_vl_std'] = np.std(cat_diff_vl_scores)
        except:
            perf_by_category[key] = {
                'true_foil_text_mean' : None,
                'true_foil_text_std' : None, 
                'active_passive_text_mean' : None,
                'active_passive_text_std' : None,
                'difference_text_mean' : None,
                'difference_text_std' : None,
                'true_foil_vl_mean' : np.mean(cat_tf_vl_scores),
                'true_foil_vl_std' : np.std(cat_tf_vl_scores), 
                'active_passive_vl_mean' : np.mean(cat_ap_vl_scores),
                'active_passive_vl_std' : np.std(cat_ap_vl_scores),
                'difference_vl_mean' : np.mean(cat_diff_vl_scores),
                'difference_vl_std' : np.std(cat_diff_vl_scores)
            }

    return (tf_text_mean,
            tf_text_std,
            ap_text_mean,
            ap_text_std,
            diff_text_mean,
            diff_text_std,
            tf_vl_mean,
            tf_vl_std,
            ap_vl_mean,
            ap_vl_std,
            diff_vl_mean,
            diff_vl_std,
            perf_by_category)