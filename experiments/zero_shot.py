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
from models.NegCLIP.negclip import CLIPWrapper
from experiments.ALBEF.eval import eval as albef_eval
from experiments.XVLM.eval import eval as xvlm_eval
from experiments.X2VLM.eval import eval as x2vlm_eval
from experiments.NegCLIP.eval import eval as negclip_eval
from experiments.BLIP.eval import eval as blip_eval
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
    parser = argparse.ArgumentParser('Set parameters for the expriments)', add_help=False)
    parser.add_argument('--model', default='X2VLM', type=str, choices=['ALBEF','XVLM','BLIP','X2VLM', 'NegCLIP'])
    parser.add_argument('--experiment', default='pre', type=str, choices=['pre', 'first_second'])
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
        'NegCLIP': load_config('../config/NegCLIP',
                             'config.yaml')

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
        'combined': configs['general']['full_dataset_path'],
        'correct_subset': configs['general']['correct_subset_path'],
        'wrong_subset': configs['general']['wrong_subset_path']
    }
    if(experiment == 'pre'):
        """ Define our dataset objects """
        ARO_correct_subset = ITMDataset(dataset_file=dataset_files['correct_subset'],
                                        dataset_name='ARO', split='active',
                                        tokenizer=tokenizer,
                                        model_name=model_name,
                                        image_preprocess=image_preprocess,
                                        model_config=configs[model_name],
                                        general_config=configs['general'])
        VALSE_correct_subset = ITMDataset(dataset_file=dataset_files['correct_subset'],
                                          dataset_name='VALSE',
                                          split='active',
                                          tokenizer=tokenizer,
                                          model_name=model_name,
                                          image_preprocess=image_preprocess,
                                          model_config=configs[model_name],
                                          general_config=configs['general'])
        ARO_wrong_subset = ITMDataset(dataset_file=dataset_files['wrong_subset'],
                                         dataset_name='ARO',
                                         split='active',
                                         tokenizer=tokenizer,
                                         model_name=model_name,
                                         image_preprocess=image_preprocess,
                                         model_config=configs[model_name],
                                         general_config=configs['general'])
        VALSE_wrong_subset = ITMDataset(dataset_file=dataset_files['wrong_subset'],
                                           dataset_name='VALSE', split='active',
                                           tokenizer=tokenizer,
                                           model_name=model_name,
                                           image_preprocess=image_preprocess,
                                           model_config=configs[model_name],
                                           general_config=configs['general'])
        """ Define our loaders """
        loaders = {
            'correct': {
                'ARO': DataLoader(ARO_correct_subset, batch_size=1, shuffle=False),
                'VALSE': DataLoader(VALSE_correct_subset, batch_size=1, shuffle=False)
            },
            'wrong': {
                'ARO': DataLoader(ARO_wrong_subset, batch_size=1, shuffle=False),
                'VALSE': DataLoader(VALSE_wrong_subset, batch_size=1, shuffle=False)
            }
        }

        _logger.info(f" Evaluation on the {dataset} benchmark - surprisal subsets. Model evaluated: {model_name}")

        """ Run the evaluation for each model """
        if(dataset == 'all' and split=='all'):
            for subset in ['correct','wrong']:
                for dataset in ['ARO','VALSE']:
                    if (model_name == 'ALBEF'):
                        acc,pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, _ = albef_eval(model,
                                                                                                                                                      loaders[subset][dataset],
                                                                                                                                                      configs['general'])
                    elif(model_name == 'XVLM'):
                        acc,pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, _ = xvlm_eval(model,
                                                                                                                                                        loaders[subset][dataset],
                                                                                                                                                        configs['general'])
                    elif (model_name == 'BLIP'):
                        acc, pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, _ = blip_eval(
                            model,
                            loaders[subset][dataset],
                            configs['general'])
                    elif (model_name == 'X2VLM'):
                        acc, pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, _ = x2vlm_eval(
                            model,
                            loaders[subset][dataset],
                            configs['general'],
                            configs['X2VLM'])
                    elif (model_name == 'NegCLIP'):
                        acc, pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, _ = negclip_eval(
                            model,
                            loaders[subset][dataset])
                    df = pd.read_csv(configs['general']['scores_'+experiment+'_path'])
                    new_row = [{
                        'model': model_name,
                        'split':subset,
                        'dataset': dataset,
                        'acc': acc,
                        'pairwise_acc': pairwise_acc,
                        'pairwise_acc_50': pairwise_acc_50,
                        'pairwise_acc_60': pairwise_acc_60,
                        'pairwise_acc_70': pairwise_acc_70,
                        'precision_caption': precision_caption,
                        'precision_foil': precision_foil
                    }]

                    row = pd.DataFrame(new_row)
                    df = pd.concat([df, row], ignore_index=True)
                    df.to_csv(configs['general']['scores_'+experiment+'_path'], index=False)

    elif(experiment == 'first_second'):
        """ Define our dataset objects """
        ARO_active_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                        dataset_name='ARO', split='active',
                                        tokenizer=tokenizer,
                                        model_name=model_name,
                                        image_preprocess=image_preprocess,
                                        model_config=configs[model_name],
                                        general_config=configs['general'])
        ARO_passive_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                         dataset_name='ARO',
                                         split='passive',
                                         tokenizer=tokenizer,
                                         model_name=model_name,
                                         image_preprocess=image_preprocess,
                                         model_config=configs[model_name],
                                         general_config=configs['general'])
        VALSE_active_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                          dataset_name='VALSE',
                                          split='active',
                                          tokenizer=tokenizer,
                                          model_name=model_name,
                                          image_preprocess=image_preprocess,
                                          model_config=configs[model_name],
                                          general_config=configs['general'])
        VALSE_passive_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                           dataset_name='VALSE', split='passive',
                                           tokenizer=tokenizer,
                                           model_name=model_name,
                                           image_preprocess=image_preprocess,
                                           model_config=configs[model_name],
                                           general_config=configs['general'])
        """ Define our loaders """
        loaders = {
            'ARO': {
                'active': DataLoader(ARO_active_dataset, batch_size=1, shuffle=False),
                'passive': DataLoader(ARO_passive_dataset, batch_size=1, shuffle=False)
            },
            'VALSE': {
                'active': DataLoader(VALSE_active_dataset, batch_size=1, shuffle=False),
                'passive': DataLoader(VALSE_passive_dataset, batch_size=1, shuffle=False)
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
                    elif (model_name == 'NegCLIP'):
                        acc, pairwise_acc, pairwise_acc_50, pairwise_acc_60, pairwise_acc_70, precision_caption, precision_foil, perf_by_cat = negclip_eval(
                            model,
                            loaders[dataset][split])
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