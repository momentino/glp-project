import torch
import torch.utils.data as data
import torch.nn as nn

from sklearn.utils import shuffle

import pandas as pd
import gdown
import os
import zipfile
from PIL import Image

from datasets.dataset_utils import preprocess_images

class ITMDataset(data.Dataset):
    """ Here for 'dataset' we mean 'VALSE' or 'ARO'.
        For 'split' we mean 'active' or 'passive'. """
    def __init__(self, dataset_file, dataset_name, split, tokenizer, model_name, model_config, general_config, image_preprocess=None):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.image_preprocess = image_preprocess
        self.model_config = model_config
        self.general_config = general_config
        self.df = self._jsonl_to_df(dataset_file)
        self.df = self.df[self.df['dataset'] == self.dataset_name] # get only the dataset we want from our merged json file

        image_file_names = self.df['image_id'].tolist()
        self.images = self._get_images(self.dataset_name, image_file_names)
        self.categories = self.df['category'].tolist()

        self.captions = self.df[self.df['dataset'] == self.dataset_name]['true_'+split].tolist()
        self.foils = self.df[self.df['dataset'] == self.dataset_name]['foil_'+split].tolist()

        self.tokenizer = tokenizer
    def _jsonl_to_df(self, file):
        df = pd.read_json(file, orient='index')
        return df

    """ Function that downloads the appropriate image folder if not found in the project, and it converts images to Python-readable objects """
    def _get_images(self, dataset_name, image_file_names):
        """ Create local image folders if they do not exist """
        image_folder = os.path.join("../datasets/images")
        if(not os.path.exists(image_folder)):
            os.mkdir(image_folder)
        dataset_img_folder = os.path.join(image_folder,dataset_name+"_images")
        if(not os.path.exists(dataset_img_folder)):
            img_url = self.general_config[dataset_name+"_image_folder_url"]
            downloaded_zip_path = os.path.join(image_folder, dataset_name + "_images.zip")
            gdown.download(img_url, output=downloaded_zip_path, fuzzy=True) # download image folder zip associated with the desired dataset
            with zipfile.ZipFile(downloaded_zip_path, 'r') as zip_ref: # unzip the downloaded file
                zip_ref.extractall(image_folder)
            os.remove(downloaded_zip_path) # delete the zip file

        image_list= []
        # open the images and save them within the list
        for f in image_file_names:
            for filename in os.listdir(dataset_img_folder):
                if(filename==f):
                    image_list.append(Image.open(os.path.join(dataset_img_folder,filename)))
        return image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if(self.image_preprocess == None):
            image = preprocess_images(config=self.model_config, model_name=self.model_name, images=image)
            caption = self.tokenizer(self.captions[idx], padding='max_length', max_length=15, truncation=True,
                                     return_tensors='pt')
            foil = self.tokenizer(self.foils[idx], padding='max_length', max_length=15, truncation=True,
                                  return_tensors='pt')
        else:
            image = self.image_preprocess(image)
            caption = self.tokenizer(self.captions[idx])
            foil = self.tokenizer(self.foils[idx])

        category = self.categories[idx]


        return image, caption, foil, category
    
#class for 3rd experiment: return image and the three needed captions
class SimilaritiesDataset(data.Dataset):
    """ Here for 'dataset' we mean 'VALSE' or 'ARO'."""
    def __init__(self, dataset_file, dataset_name, tokenizer, general_config, model_name, model_config):
        self.general_config = general_config
        self.model_config = model_config
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.df = self._jsonl_to_df(dataset_file)
        self.df = self.df[self.df['dataset'] == self.dataset_name] # get only the dataset we want from our merged json file

        image_file_names = self.df['image_id'].tolist()
        self.images = self._get_images(self.dataset_name, image_file_names)
        self.categories = self.df['category'].tolist()

        self.true_actives = self.df[self.df['dataset'] == self.dataset_name]['true_active'].tolist()
        self.foil_actives = self.df[self.df['dataset'] == self.dataset_name]['foil_active'].tolist()
        self.true_passives = self.df[self.df['dataset'] == self.dataset_name]['true_passive'].tolist()

        self.tokenizer = tokenizer
    def _jsonl_to_df(self, file):
        df = pd.read_json(file, orient='index')
        return df

    """ Function that downloads the appropriate image folder if not found in the project, and it converts images to Python-readable objects """
    def _get_images(self, dataset_name, image_file_names):
        """ Create local image folders if they do not exist """
        image_folder = os.path.join("../datasets/images")
        if(not os.path.exists(image_folder)):
            os.mkdir(image_folder)
        dataset_img_folder = os.path.join(image_folder,dataset_name+"_images")
        if(not os.path.exists(dataset_img_folder)):
            img_url = self.general_config[dataset_name+"_image_folder_url"]
            downloaded_zip_path = os.path.join(image_folder, dataset_name + "_images.zip")
            gdown.download(img_url, output=downloaded_zip_path, fuzzy=True) # download image folder zip associated with the desired dataset
            with zipfile.ZipFile(downloaded_zip_path, 'r') as zip_ref: # unzip the downloaded file
                zip_ref.extractall(image_folder)
            os.remove(downloaded_zip_path) # delete the zip file

        image_list= []
        # open the images and save them within the list
        for f in image_file_names:
            for filename in os.listdir(dataset_img_folder):
                if(filename==f):
                    image_list.append(Image.open(os.path.join(dataset_img_folder,filename)))
        return image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = preprocess_images(config=self.model_config, model_name=self.model_name, images=image)
        category = self.categories[idx]
        true_active = self.tokenizer(self.true_actives[idx], padding='max_length', max_length=15, truncation=True, return_tensors='pt')
        foil_active = self.tokenizer(self.foil_actives[idx], padding='max_length', max_length=15, truncation=True, return_tensors='pt')
        true_passive = self.tokenizer(self.true_passives[idx], padding='max_length', max_length=15, truncation=True, return_tensors='pt')

        return image, true_active, foil_active, true_passive, category

