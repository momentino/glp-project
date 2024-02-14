import logging
import argparse
import sys
import os
import yaml
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from datasets.datasets import DistancesDataset
from models.ALBEF.models.model_pretrain import ALBEF
from models.XVLM.models.model_pretrain import XVLM as XVLM
from models.X2VLM.models.model_pretrain import XVLM as X2VLM
from models.BLIP.models.blip_pretrain import BLIP_Pretrain
from experiments.ALBEF.similarities import similarities as albef_similarities
from experiments.XVLM.eval import eval as xvlm_eval
from experiments.X2VLM.eval import eval as x2vlm_eval
from experiments.BLIP.eval import eval as blip_eval
from utils.utils import download_weights

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
    parser.add_argument('--model', default='X2VLM', type=str, choices=['ALBEF','XVLM','BLIP','X2VLM'])
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
    }

    # our tokenizer is initialized from the text encoder specified in the config file
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
    elif(model_name == 'BLIP'):
        model = BLIP_Pretrain(image_size=configs['BLIP']['image_res'], vit=configs['BLIP']['vit'],
                      vit_grad_ckpt=configs['BLIP']['vit_grad_ckpt'],
                      vit_ckpt_layer=configs['BLIP']['vit_ckpt_layer'], queue_size=configs['BLIP']['queue_size'],
                      med_config=configs['BLIP']['bert_config'])
    elif(model_name == 'XVLM'):
        model = XVLM(config=configs['XVLM'])
    elif(model_name == 'X2VLM'):
        model = X2VLM(config=configs['X2VLM'], load_text_params=True, load_vision_params=True, pretraining=False)


    dataset_files = {
        'combined': configs['general']['full_dataset_path']
    }
    
    """ Define our dataset objects """
    ARO_dataset = DistancesDataset(dataset_file=dataset_files['combined'],
                                    dataset_name='ARO',
                                    tokenizer=tokenizer,
                                    general_config=configs['general'],
                                    model_name=model_name,
                                    model_config=configs[model_name]
                                    )
    VALSE_dataset = DistancesDataset(dataset_file=dataset_files['combined'],
                                        dataset_name='VALSE',
                                        tokenizer=tokenizer,
                                        general_config=configs['general'],
                                        model_name=model_name,
                                        model_config=configs[model_name]
                                        )

    """ Define our loaders """
    loaders = {
            'ARO': DataLoader(ARO_dataset, batch_size=64, shuffle=False),
            'VALSE': DataLoader(VALSE_dataset, batch_size=64, shuffle=False)
        }

    _logger.info(f" Evaluation on the {dataset} benchmark. Model evaluated: {model_name}")

    """ Run the evaluation for each model """
    if(dataset == 'all'):
        for dataset in ['ARO','VALSE']:
            if (model_name == 'ALBEF'):
                tf_mean, tf_std, ap_mean, ap_std, diff_mean, diff_std, perf_by_cat = albef_similarities(model,loaders[dataset],configs['general'])
                
            elif(model_name == 'XVLM'):
                pass    
            elif (model_name == 'BLIP'):
                pass    
            elif (model_name == 'X2VLM'):
                pass    
            df = pd.read_csv(configs['general']['scores_'+experiment+'_path'])
            rows = []
            new_row = {
                        'model': model_name,
                        'dataset': dataset,
                        'category': None, # because this is the row with the general results as we want in the pre and first experiments
                        'true_foil_mean':tf_mean,
                        'true_foil_std':tf_std,
                        'active_passive_mean':ap_mean,
                        'active_passive_std':ap_std,
                        'difference_mean':diff_mean,
                        'difference_std':diff_std
                    }
            rows.append(new_row)
           
            for key,value in perf_by_cat.items():
                new_row = {
                            'model': model_name,
                            'dataset': dataset,
                            'category': key,
                            # because this is the row with the general results as we want in the pre and first experiments
                            'true_foil_mean':value['true_foil_mean'],
                            'true_foil_std':value['true_foil_std'],
                            'active_passive_mean':value['active_passive_mean'],
                            'active_passive_std':value['active_passive_std'],
                            'difference_mean':value['difference_mean'],
                            'difference_std':value['difference_std']
                            }
                rows.append(new_row)
            rows = pd.DataFrame(rows)
            df = pd.concat([df, rows], ignore_index=True)
            df.to_csv(configs['general']['scores_'+experiment+'_path'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GLP Project', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)