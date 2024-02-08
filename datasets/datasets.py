import torch
import torch.utils.data as data
import torch.nn as nn

from sklearn.utils import shuffle

import pandas as pd
import gdown
import os
import zipfile
from PIL import Image

class ITMDataset(data.Dataset):
    """ Here for 'dataset' we mean 'VALSE' or 'ARO'.
        For 'split' we mean 'active' or 'passive'. """
    def __init__(self, dataset_file, dataset_name, split, tokenizer, config):
        self.config = config
        self.dataset_name = dataset_name
        self.df = self._jsonl_to_df(dataset_file, random_seed=42)
        self.df = self.df[self.df['dataset'] == self.dataset_name] # get only the dataset we want from our merged json file

        self.images = self._get_images(self.dataset_name)
        self.categories = self.df['categories'].to_list()

        self.captions = self.df[self.df['true_'+split] == self.dataset_name].to_list()
        self.foils = self.df[self.df['true_'+split] == self.dataset_name].to_list()

        self.tokenizer = tokenizer
    def _jsonl_to_df(self, file, random_seed):
        df = pd.read_json(file, orient='index')
        df = shuffle(df, random_state=random_seed)
        return df

    """ Function that downloads the appropriate image folder if not found in the project, and it converts images to Python-readable objects """
    def _get_images(self, dataset_name):
        """ Create local image folders if they do not exist """
        image_folder = os.path.join("datasets/images")
        if(not os.path.exists(image_folder)):
            os.mkdir(image_folder)
        dataset_img_folder = os.path.join(image_folder,dataset_name+"_images")
        if(not os.path.exists(dataset_img_folder)):
            img_url = self.config[dataset_name+"_image_folder_url"]
            os.mkdir(dataset_img_folder)
            gdown.download(img_url, output=dataset_img_folder, quiet=False) # download image folder zip associated with the desired dataset
            downloaded_zip_path = os.path.join(dataset_img_folder,dataset_name+"_images.zip")
            with zipfile.ZipFile(downloaded_zip_path, 'r') as zip_ref: # unzip the downloaded file
                zip_ref.extractall(dataset_img_folder)
            os.remove(downloaded_zip_path) # delete the zip file

        image_list= []
        # open the images and save them within the list
        for filename in os.listdir(dataset_img_folder):
            image_list.append(Image.open(os.path.join(dataset_img_folder,filename)))

        return  image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx])
        category = self.categories[idx]
        caption = self.tokenizer(self.captions[idx], padding=True, max_length=512, truncation=True, return_tensors='pt')
        foil = self.tokenizer(self.foils[idx], padding=True, max_length=512, truncation=True, return_tensors='pt')

        return image, caption, foil, category

