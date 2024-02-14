import logging
import argparse
import sys
import os
import yaml
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from datasets.datasets import ITMDataset
from models.ALBEF.models.model_pretrain import ALBEF
from models.XVLM.models.model_pretrain import XVLM as XVLM
from models.X2VLM.models.model_pretrain import XVLM as X2VLM
from models.BLIP.models.blip_pretrain import BLIP_Pretrain
from experiments.ALBEF.eval import eval as albef_eval
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
    parser.add_argument('--experiment', default='first_second', type=str, choices=['pre', 'first_second'])
    parser.add_argument('--dataset', default='all', type=str, choices=['VALSE', 'ARO','all'])
    parser.add_argument('--split', default='all', type=str, choices=['active', 'passive','all'])

    return parser

# Function to load yaml configuration file
def load_config(config_path, config_name):
    with open(os.path.join(config_path, config_name)) as file:
        config = yaml.safe_load(file)

    return config

def main(args):
    model_name = args.model
    experiment = args.experiment
    dataset = args.dataset
    split = args.split


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
    if(experiment == 'pre'):
        #TODO
        pass
    elif(experiment == 'first_second'):
        """ Define our dataset objects """
        ARO_active_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                        dataset_name='ARO', split='active',
                                        tokenizer=tokenizer,
                                        general_config=configs['general'],
                                        model_name=model_name,
                                        model_config=configs[model_name])
        ARO_passive_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                         dataset_name='ARO',
                                         split='passive',
                                         tokenizer=tokenizer,
                                         general_config=configs['general'],
                                         model_name=model_name,
                                         model_config=configs[model_name]
                                         )
        VALSE_active_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                          dataset_name='VALSE',
                                          split='active',
                                          tokenizer=tokenizer,
                                          general_config=configs['general'],
                                          model_name=model_name,
                                          model_config=configs[model_name]
                                          )
        VALSE_passive_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                           dataset_name='VALSE', split='passive',
                                           tokenizer=tokenizer,
                                           general_config=configs['general'],
                                           model_name=model_name,
                                           model_config=configs[model_name]
                                           )
        """ Define our loaders """
        loaders = {
            'ARO': {
                'active': DataLoader(ARO_active_dataset, batch_size=64, shuffle=False),
                'passive': DataLoader(ARO_passive_dataset, batch_size=64, shuffle=False)
            },
            'VALSE': {
                'active': DataLoader(VALSE_active_dataset, batch_size=64, shuffle=False),
                'passive': DataLoader(VALSE_passive_dataset, batch_size=64, shuffle=False)
            }
        }

        _logger.info(f" Evaluation on the {dataset} benchmark - {split} mode. Model evaluated: {model_name}")

        """ Run the evaluation for each model """
        if(dataset == 'all' and split=='all'):
            for dataset in ['ARO','VALSE']:
                for split in ['active','passive']:
                    if (model_name == 'ALBEF'):
                        acc,pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, perf_by_cat = albef_eval(model,
                                                                                                                                                      loaders[dataset][split],
                                                                                                                                                      configs['general'])
                    elif(model_name == 'XVLM'):
                        acc,pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, perf_by_cat = xvlm_eval(model,
                                                                                                                                                        loaders[dataset][split],
                                                                                                                                                        configs['general'])
                    elif (model_name == 'BLIP'):
                        acc, pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, perf_by_cat = blip_eval(
                            model,
                            loaders[dataset][split],
                            configs['general'])
                    elif (model_name == 'X2VLM'):
                        acc, pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, perf_by_cat = x2vlm_eval(
                            model,
                            loaders[dataset][split],
                            configs['general'],
                            configs['X2VLM'])
                    df = pd.read_csv(configs['general']['scores_'+experiment+'_path'])
                    rows = []
                    new_row = {
                        'model': model_name,
                        'dataset': dataset,
                        'split': split,
                        'category': None, # because this is the row with the general results as we want in the pre and first experiments
                        'acc': acc,
                        'pairwise_acc': pairwise_acc,
                        'pairwise_acc_50': pairwise_acc_50,
                        'pairwise_acc_60': pairwise_acc_60,
                        'pairwise_acc_70': pairwise_acc_70,
                        'precision_caption': precision_caption,
                        'precision_foil': precision_foil


                    }
                    rows.append(new_row)
                    if( experiment == 'first_second'):
                        for key,value in perf_by_cat.items():
                            new_row = {
                                'model': model_name,
                                'dataset': dataset,
                                'split': split,
                                'category': key,
                                # because this is the row with the general results as we want in the pre and first experiments
                                'acc': value['acc'],
                                'pairwise_acc': value['pairwise_acc'],
                                'pairwise_acc_50': value['pairwise_acc_50'],
                                'pairwise_acc_60': value['pairwise_acc_60'],
                                'pairwise_acc_70': value['pairwise_acc_70'],
                                'precision_caption': value['precision_caption'],
                                'precision_foil': value['precision_foil']

                            }
                            rows.append(new_row)
                    rows = pd.DataFrame(rows)
                    df = pd.concat([df, rows], ignore_index=True)
                    df.to_csv(configs['general']['scores_'+experiment+'_path'], index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GLP Project', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)