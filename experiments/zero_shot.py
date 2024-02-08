import logging
import argparse
import sys
import os
import yaml
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from datasets.datasets import ITMDataset
from models.ALBEF.models.model_pretrain import ALBEF
from experiments.ALBEF.eval import eval as albef_eval

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
    parser.add_argument('--model', default='ALBEF', type=str, choices=['ALBEF','XVLM','BLIP','CLIP','NegCLIP'])
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
        'general': load_config('config/general',
                         'general_config.yaml'),  # load the configuration file (the parameters will then be used like a dictionary with key-value pairs
        'ALBEF': load_config('config/ALBEF',
                         'config.yaml'),
    }

    # our tokenizer is initialized from the text encoder specified in the config file
    tokenizer = AutoTokenizer.from_pretrained(configs[model_name]['text_encoder'])


    models = {
        'ALBEF': ALBEF(text_encoder=configs['ALBEF']['text_encoder'], tokenizer=tokenizer)
    }

    # load the model
    model = models[model_name]

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
                                        config=configs['general'])
        ARO_passive_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                         dataset_name='ARO',
                                         split='passive',
                                         tokenizer=tokenizer,
                                         config=configs['general'])
        VALSE_active_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                          dataset_name='VALSE',
                                          split='active',
                                          tokenizer=tokenizer,
                                          config=configs['general'])
        VALSE_passive_dataset = ITMDataset(dataset_file=dataset_files['combined'],
                                           dataset_name='VALSE', split='passive',
                                           tokenizer=tokenizer,
                                           config=configs['general'])
        """ Define our loaders """
        ARO_active_loader = DataLoader(ARO_active_dataset, batch_size=64, shuffle=True)
        ARO_passive_loader = DataLoader(ARO_passive_dataset, batch_size=64, shuffle=True)
        VALSE_active_loader = DataLoader(VALSE_active_dataset, batch_size=64, shuffle=True)
        VALSE_passive_loader = DataLoader(VALSE_passive_dataset, batch_size=64, shuffle=True)

        _logger.info(f" Evaluation on the {dataset} benchmark - {split} mode. Model evaluated: {model_name}")

        """ Run the evaluation for each model """

        performances = {
            'ARO': {
                'active': dict(),
                'passive': dict()
            },
            'VALSE': {
                'active': dict(),
                'passive': dict()
            }
        }
        if(model_name == 'ALBEF'):
            if(dataset == 'all' and split=='all'):
                pairwise_acc, acc, true_prec, foil_prec, rec, perf_by_cat = albef_eval(model, ARO_active_loader) # TODO
                performances['ARO']['active'] = {'pairwise_acc': pairwise_acc,
                                                 'acc': acc,
                                                 'true_prec':true_prec,
                                                 'foil_prec': foil_prec,
                                                 'rec': rec,
                                                 'perf_by_cat': perf_by_cat}
                pairwise_acc, acc, true_prec, foil_prec, rec, perf_by_cat = albef_eval(model, ARO_passive_loader)  # TODO
                performances['ARO']['passive'] = {'pairwise_acc': pairwise_acc,
                                                  'acc': acc,
                                                  'true_prec': true_prec,
                                                  'foil_prec': foil_prec,
                                                  'rec': rec,
                                                  'perf_by_cat': perf_by_cat}
                pairwise_acc, acc, true_prec, foil_prec, rec, perf_by_cat = albef_eval(model,
                                                                                       VALSE_active_loader)  # TODO
                performances['VALSE']['active'] = {'pairwise_acc': pairwise_acc,
                                                   'acc': acc,
                                                   'true_prec': true_prec,
                                                   'foil_prec': foil_prec,
                                                   'rec': rec,
                                                   'perf_by_cat': perf_by_cat}
                pairwise_acc, acc, true_prec, foil_prec, rec, perf_by_cat = albef_eval(model,
                                                                                       VALSE_passive_loader)  # TODO
                performances['VALSE']['passive'] = {'pairwise_acc': pairwise_acc,
                                                    'acc': acc,
                                                    'true_prec': true_prec,
                                                    'foil_prec': foil_prec,
                                                    'rec': rec,
                                                    'perf_by_cat': perf_by_cat}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('GLP Project', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)