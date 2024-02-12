from tqdm import tqdm
from models.AdaptedModels.XVLMForITM import XVLMForITM
import torch
from utils.utils import load_weights
def eval(model, loader, config):
    adapted_model = XVLMForITM(model)
    load_weights(adapted_model.base_model, model_name='XVLM', general_config=config) # we load the weights of the base architecture
    adapted_model.eval()

    c_scores = []
    f_scores = []
    total_num_samples = 0
    with torch.no_grad():
        for images, captions, foils, categories in tqdm(loader):
            caption_scores, foils_scores = adapted_model(images, captions, foils)
            c_scores.extend(caption_scores)
            f_scores.extend(foils_scores)
            total_num_samples += 64

            print(sum([1 if c_scores[i].item() > f_scores[i].item() else 0 for i in
                       range(len(c_scores))]) / total_num_samples)
    pairwise_acc = sum(
        [1 if c_scores[i].item() > f_scores[i].item() else 0 for i in range(len(c_scores))]) / total_num_samples
    pairwise_acc_50 = sum([1 if c_scores[i].item() > f_scores[i].item() and c_scores[i].item() > 0.5 else 0 for i in
                           range(len(c_scores))]) / total_num_samples
    pairwise_acc_60 = sum([1 if c_scores[i].item() > f_scores[i].item() and c_scores[i].item() > 0.6 else 0 for i in
                           range(len(c_scores))]) / total_num_samples
    pairwise_acc_70 = sum([1 if c_scores[i].item() > f_scores[i].item() and c_scores[i].item() > 0.7 else 0 for i in
                           range(len(c_scores))]) / total_num_samples
    print(pairwise_acc, " ", pairwise_acc_50, " ", pairwise_acc_60, " ", pairwise_acc_70)