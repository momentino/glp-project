import gdown
import os
import zipfile
import torch

def download_weights(model_name, general_config):
    weights_folder = "../pretrained_weights"
    if (not os.path.exists(weights_folder)):
        os.mkdir(weights_folder)
    model_weights_file = [os.path.join(weights_folder, model_name+"_weights.pth"),os.path.join(weights_folder, model_name+"_weights.th")]
    if (not os.path.exists(model_weights_file[0]) and not os.path.exists(model_weights_file[1])):
        weights_url = general_config[model_name+"_weights"]
        downloaded_zip_path = os.path.join(weights_folder, model_name + "_weights.pth")
        gdown.download(weights_url, output=downloaded_zip_path,
                       fuzzy=True)  # download image folder zip associated with the desired dataset

def load_weights(model, model_name, general_config):
    path = '../pretrained_weights/'+model_name+"_weights.pth"
    if (not os.path.exists(path)):
        download_weights(model_name, general_config)
    model.load_state_dict(torch.load(path), strict=False)