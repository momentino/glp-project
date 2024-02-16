from tqdm import tqdm

import torch
import torch.nn.functional as F
import numpy as np

def similarities(model, loader):
    model.model.eval()

    tf_text_scores=[]
    ap_text_scores=[]
    diff_text_scores=[]

    scores_by_cat = dict()
   
    with torch.no_grad():
        for images, true_actives, foil_actives, true_passives, categories in tqdm(loader):
            for image, true_active, foil_active, true_passive, cat in zip(images, true_actives, foil_actives, true_passives, categories):

                ta_features = model.model.encode_text(true_active)
                fa_features = model.model.encode_text(foil_active)
                tp_features = model.model.encode_text(true_passive)

                tf_text_similarities=F.cosine_similarity(ta_features,fa_features,dim=-1)
                ap_text_similarities=F.cosine_similarity(ta_features,tp_features,dim=-1)

                tf_text_scores.extend(tf_text_similarities)
                ap_text_scores.extend(ap_text_similarities)
                diff_text_scores.extend(tf_text_similarities-ap_text_similarities)
                  
                for cat, tf_t_sc,ap_t_sc,d_t_sc in zip(categories, tf_text_scores, ap_text_scores, diff_text_scores):
                    try:
                        scores_by_cat[cat]['true_foil_text_scores'].append(tf_t_sc)
                        scores_by_cat[cat]['active_passive_text_scores'].append(ap_t_sc)
                        scores_by_cat[cat]['difference_text_scores'].append(d_t_sc)
                    except:
                        scores_by_cat[cat] = {
                            'true_foil_text_scores': [tf_t_sc],
                            'active_passive_text_scores': [ap_t_sc],
                            'difference_text_scores':[d_t_sc]
                            }
    tf_text_mean=np.mean(tf_text_scores)
    tf_text_std=np.std(tf_text_scores)
    ap_text_mean=np.mean(ap_text_scores)
    ap_text_std=np.std(ap_text_scores)
    diff_text_mean=np.mean(diff_text_scores)
    diff_text_std=np.std(diff_text_scores)

    tf_vl_mean=None #for the csv
    tf_vl_std=None
    ap_vl_mean=None
    ap_vl_std=None
    diff_vl_mean=None
    diff_vl_std=None

    perf_by_category = {}
    for key, value in scores_by_cat.items():
        cat_tf_t_scores = value['true_foil_text_scores']
        cat_ap_t_scores = value['active_passive_text_scores']
        cat_diff_t_scores = value['difference_text_scores']
            
        try:
            perf_by_category[key]['true_foil_text_mean'] = np.mean(cat_tf_t_scores)
            perf_by_category[key]['true_foil_text_std'] = np.std(cat_tf_t_scores)
            perf_by_category[key]['active_passive_text_mean'] = np.mean(cat_ap_t_scores)
            perf_by_category[key]['active_passive_text_std'] = np.std(cat_ap_t_scores)
            perf_by_category[key]['difference_text_mean'] = np.mean(cat_diff_t_scores)
            perf_by_category[key]['difference_text_std'] = np.std(cat_diff_t_scores)

            perf_by_category[key]['true_foil_vl_mean'] = None
            perf_by_category[key]['true_foil_vl_std'] = None
            perf_by_category[key]['active_passive_vl_mean'] = None
            perf_by_category[key]['active_passive_vl_std'] = None
            perf_by_category[key]['difference_vl_mean'] = None
            perf_by_category[key]['difference_vl_std'] = None
        except:
            perf_by_category[key] = {
                'true_foil_text_mean' : np.mean(cat_tf_t_scores),
                'true_foil_text_std' : np.std(cat_tf_t_scores), 
                'active_passive_text_mean' : np.mean(cat_ap_t_scores),
                'active_passive_text_std' : np.std(cat_ap_t_scores),
                'difference_text_mean' : np.mean(cat_diff_t_scores),
                'difference_text_std' : np.std(cat_diff_t_scores),
                'true_foil_vl_mean' : None,
                'true_foil_vl_std' : None, 
                'active_passive_vl_mean' : None,
                'active_passive_vl_std' : None,
                'difference_vl_mean' : None,
                'difference_vl_std' : None
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