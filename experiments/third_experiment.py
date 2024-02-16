import logging
import argparse
import sys
import os
import yaml
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from datasets.datasets import SimilaritiesDataset
from models.ALBEF.models.model_pretrain import ALBEF
from models.XVLM.models.model_pretrain import XVLM as XVLM
from models.X2VLM.models.model_pretrain import XVLM as X2VLM
from models.BLIP.models.blip_pretrain import BLIP_Pretrain
from models.NegCLIP.negclip import CLIPWrapper
from experiments.ALBEF.similarities import similarities as albef_similarities
from experiments.XVLM.similarities import similarities as xvlm_similarities
from experiments.X2VLM.similarities import similarities as x2vlm_similarities
from experiments.BLIP.similarities import similarities as blip_similarities
from experiments.NegCLIP.similarities import similarities as negclip_similarities
from utils.utils import download_weights
import open_clip

_logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--log_level', type=str, default='INFO')
FLAGS, FIRE_FLAGS = parser.parse_known_args()
logging.basicConfig(stream=sys.stdout, level=logging.getLevelName(FLAGS.log_level))
_logger.info(f"Running with args {FLAGS}, {FIRE_FLAGS}")


""" 
    Our arguments to set our experiments. You may set them from command line when we execute the file.
    However, you may also just change the default value here every time. 
    
"""
def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for the expriments', add_help=False)
    parser.add_argument('--model', default='BLIP', type=str, choices=['ALBEF','XVLM','BLIP','X2VLM','NegCLIP'])
    parser.add_argument('--experiment', default='third', type=str, choices=['third'])
    parser.add_argument('--dataset', default='all', type=str, choices=['VALSE', 'ARO','all'])

    return parser

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def main(args):
    model_name = args.model
    dataset = args.dataset
    experiment = args.experiment

    configs = {
        'general': load_config('../config/general',
                         'general_config.yaml'),  # load the configuration file (the parameters will then be used like a dictionary with key-value pairs
        'ALBEF': load_config('../config/ALBEF',
                         'config.yaml'),
        'XVLM': load_config('../config/XVLM',
                             'config.yaml'),
        'BLIP': load_config('../config/BLIP',
                             'config.yaml'),
        'X2VLM': load_config('../config/X2VLM',
                             'config.yaml'),
        'NegCLIP': load_config('../config/NegCLIP',
                             'config.yaml'),
    }

    # our tokenizer is initialized from the text encoder specified in the config file
    if(model_name=='NegCLIP'):
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
    else:
        tokenizer = AutoTokenizer.from_pretrained(configs[model_name]['text_encoder'])

    if(model_name=='XVLM'):
        download_weights(model_name='swin',
                         general_config=configs['general'])  # to download the vision encoder weights if not done already
    if(model_name=='X2VLM'):
        download_weights(model_name='beitv2_base_patch16_224_pt1k_ft21k',
                         general_config=configs[
                             'general'])  # to download the vision encoder weights if not done already
    # load the model
    if(model_name == 'ALBEF'):
        model= ALBEF(config=configs['ALBEF'], text_encoder=configs['ALBEF']['text_encoder'], tokenizer=tokenizer)
        image_preprocess = None
    elif(model_name == 'BLIP'):
        model = BLIP_Pretrain(image_size=configs['BLIP']['image_res'], vit=configs['BLIP']['vit'],
                      vit_grad_ckpt=configs['BLIP']['vit_grad_ckpt'],
                      vit_ckpt_layer=configs['BLIP']['vit_ckpt_layer'], queue_size=configs['BLIP']['queue_size'],
                      med_config=configs['BLIP']['bert_config'])
        image_preprocess = None
    elif(model_name == 'XVLM'):
        model = XVLM(config=configs['XVLM'])
        image_preprocess = None
    elif(model_name == 'X2VLM'):
        model = X2VLM(config=configs['X2VLM'], load_text_params=True, load_vision_params=True, pretraining=False)
        image_preprocess = None
    elif(model_name=='NegCLIP'):
        path = os.path.join('../pretrained_weights', "NegCLIP_weights.pth")
        if not os.path.exists(path):
            print("Downloading the NegCLIP model...")
            import gdown
            gdown.download(id="1ooVVPxB-tvptgmHlIMMFGV3Cg-IrhbRZ", output=path, quiet=False)
        model, _, image_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained=path, device='cpu')
        model = CLIPWrapper(model, 'cpu')


    dataset_files = {
        'combined': configs['general']['full_dataset_path']
    }
    
    """ Define our dataset objects """
    ARO_dataset = SimilaritiesDataset(dataset_file=dataset_files['combined'],
                                    dataset_name='ARO',
                                    tokenizer=tokenizer,
                                    general_config=configs['general'],
                                    model_name=model_name,
                                    model_config=configs[model_name],
                                    image_preprocess=image_preprocess
                                    )
    VALSE_dataset = SimilaritiesDataset(dataset_file=dataset_files['combined'],
                                        dataset_name='VALSE',
                                        tokenizer=tokenizer,
                                        general_config=configs['general'],
                                        model_name=model_name,
                                        model_config=configs[model_name],
                                        image_preprocess=image_preprocess
                                        )

    """ Define our loaders """
    loaders = {
            'ARO': DataLoader(ARO_dataset, batch_size=1, shuffle=False),
            'VALSE': DataLoader(VALSE_dataset, batch_size=1, shuffle=False)
        }

    _logger.info(f" Evaluation on the {dataset} benchmark. Model evaluated: {model_name}")

    """ Run the evaluation for each model """
    if(dataset == 'all'):
        for dataset in ['ARO','VALSE']:
            if (model_name == 'ALBEF'):
                tf_t_mean, tf_t_std, ap_t_mean, ap_t_std, diff_t_mean, diff_t_std, tf_vl_mean, tf_vl_std, ap_vl_mean, ap_vl_std, diff_vl_mean, diff_vl_std, perf_by_cat = albef_similarities(model,loaders[dataset],configs['general'])
                
            elif(model_name == 'XVLM'):
                tf_t_mean, tf_t_std, ap_t_mean, ap_t_std, diff_t_mean, diff_t_std, tf_vl_mean, tf_vl_std, ap_vl_mean, ap_vl_std, diff_vl_mean, diff_vl_std, perf_by_cat = xvlm_similarities(model,loaders[dataset],configs['general'])    
            
            elif (model_name == 'BLIP'):
                tf_t_mean, tf_t_std, ap_t_mean, ap_t_std, diff_t_mean, diff_t_std, tf_vl_mean, tf_vl_std, ap_vl_mean, ap_vl_std, diff_vl_mean, diff_vl_std, perf_by_cat = blip_similarities(model,loaders[dataset],configs['general'])    
            
            elif (model_name == 'X2VLM'):
                tf_t_mean, tf_t_std, ap_t_mean, ap_t_std, diff_t_mean, diff_t_std, tf_vl_mean, tf_vl_std, ap_vl_mean, ap_vl_std, diff_vl_mean, diff_vl_std, perf_by_cat = x2vlm_similarities(model,loaders[dataset],configs['general'],configs['X2VLM'])    
            
            elif (model_name == 'NegCLIP'):
                tf_t_mean, tf_t_std, ap_t_mean, ap_t_std, diff_t_mean, diff_t_std, tf_vl_mean, tf_vl_std, ap_vl_mean, ap_vl_std, diff_vl_mean, diff_vl_std, perf_by_cat = negclip_similarities(model,loaders[dataset])    

            df = pd.read_csv(configs['general']['scores_'+experiment+'_path'])
            rows = []
            new_row = {
                        'model': model_name,
                        'dataset': dataset,
                        'category': None, # because this is the row with the general results as we want in the pre and first experiments
                        'true_foil_text_mean':tf_t_mean,
                        'true_foil_text_std':tf_t_std,
                        'active_passive_text_mean':ap_t_mean,
                        'active_passive_text_std':ap_t_std,
                        'difference_text_mean':diff_t_mean,
                        'difference_text_std':diff_t_std,
                        'true_foil_vl_mean':tf_vl_mean,
                        'true_foil_vl_std':tf_vl_std,
                        'active_passive_vl_mean':ap_vl_mean,
                        'active_passive_vl_std':ap_vl_std,
                        'difference_vl_mean':diff_vl_mean,
                        'difference_vl_std':diff_vl_std
                    }
            rows.append(new_row)
           
            for key,value in perf_by_cat.items():
                new_row = {
                            'model': model_name,
                            'dataset': dataset,
                            'category': key,
                            'true_foil_text_mean':value['true_foil_text_mean'],
                            'true_foil_text_std':value['true_foil_text_std'],
                            'active_passive_text_mean':value['active_passive_text_mean'],
                            'active_passive_text_std':value['active_passive_text_std'],
                            'difference_text_mean':value['difference_text_mean'],
                            'difference_text_std':value['difference_text_std'],
                            'true_foil_vl_mean':value['true_foil_vl_mean'],
                            'true_foil_vl_std':value['true_foil_vl_std'],
                            'active_passive_vl_mean':value['active_passive_vl_mean'],
                            'active_passive_vl_std':value['active_passive_vl_std'],
                            'difference_vl_mean':value['difference_vl_mean'],
                            'difference_vl_std':value['difference_vl_std']
                            }
                rows.append(new_row)
            rows = pd.DataFrame(rows)
            df = pd.concat([df, rows], ignore_index=True)
            df.to_csv(configs['general']['scores_'+experiment+'_path'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GLP Project', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)