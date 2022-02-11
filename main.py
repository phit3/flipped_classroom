#!/usr/bin/python3
import argparse
import re
import json

from model_manager import ModelManager
from models.tf import TF
from models.fr import FR
from models.cl_ctf_p import CL_CTF_P
from models.cl_dtf import CL_DTF_P_Lin, CL_DTF_P_InvSig, CL_DTF_P_Exp, CL_DTF_D_Lin, CL_DTF_D_InvSig, CL_DTF_D_Exp
from models.cl_itf import CL_ITF_P_Lin, CL_ITF_P_InvSig, CL_ITF_P_Exp, CL_ITF_D_Lin, CL_ITF_D_InvSig, CL_ITF_D_Exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replication package for: Flipped Classroom: Effective Teaching for Chaotic Time Series Forecasting.')
    parser.add_argument('-m', '--models', dest='model_class_name', action='extend', nargs='+', default=[],
                        help='class name(s) of the model(s) to train or test.')
    parser.add_argument('-o', '--operation', dest='operation', default='test', choices=['train', 'test'], help='operation that shall be performed.')

    parser.add_argument('-d', '--datasets', dest='dataset', action='extend', nargs='+', default=[],
                        help='datasets to be processed optionally together with the desired hyperparameters in JSON format (e.g. \'{"thomas_0.1_0.055": {"gamma": 0.6, "lr": 1e-3, "plateau": 10, "latent_dim": 256, "output_steps": 182}}\').')
    parser.add_argument('-t', '--tag', dest='tag', type=str, default='test', help='tag that identifies a set of experiments')
    parser.add_argument('-q', '--quiet', dest='quiet', default=False, action='store_true', help='reduces noise on your command line.')

    args = parser.parse_args()
    operation = args.operation
    tag = args.tag
    quiet = args.quiet

    hyperparameter_sets = {'hyperroessler_0.1_0.14':
                               {'dimensions': 4,
                                'lr': 3e-5,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 72},
                           'lorenz96_0.05_1.67':
                               {'dimensions': 40,
                                'lr': 1e-3,
                                'gamma': 0.9,
                                'plateau': 20,
                                'latent_dim': 256,
                                'output_steps': 12},
                           'lorenz_0.01_0.905':
                               {'dimensions': 3,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 111},
                           'mackeyglass_1.0_0.006':
                               {'dimensions': 1,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 167},
                           'roessler_0.12_0.069':
                               {'dimensions': 3,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 121},
                           'thomas_0.1_0.055':
                               {'dimensions': 3,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 182},
                           'default':
                               {'dimensions': 3,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 200},
                           }
    datasets = {}
    for dataset in args.dataset:
        if dataset.startswith('"') or dataset.startswith('{') or dataset.startswith('['):
            dataset = json.loads(dataset)
            if type(dataset) is str:
                datasets[dataset] = {}
            elif type(dataset) is list:
                for d in dataset:
                    datasets[d] = {}
            elif type(dataset) is dict:
                for d, v in dataset.items():
                    datasets[d] = v
            else:
                print('Datasets information were not provided adequatly.')
        else:
            datasets[dataset] = {}

    for dataset, override_args in datasets.items():
        dataset = re.sub(r'.csv$', '', dataset)
        if not override_args:
            override_args = {}
        if dataset in hyperparameter_sets:
            hyperparameters = hyperparameter_sets[dataset]
        else:
            print(f'[WARNING] Did not find any predefined hyperparameters for {dataset} - using default set as basis.')
            hyperparameters = hyperparameter_sets['default']

        for hp, new in override_args.items():
            if hp in hyperparameters:
                old = hyperparameters[hp]
                if not quiet:
                    print(f'Overriding {hp}: {old} -> {new}.')
                hyperparameters[hp] = new

        for model_name in args.model_class_name:
            try:
                model_class = eval(model_name)
                if not quiet:
                    print('Running {} {} operation for {} on {}.'. format(tag, operation, model_name, dataset))
                mgr = ModelManager(model_class=model_class, dataset=dataset, tag=tag, quiet=quiet)
                if operation == 'train':
                    mgr.train_model(override_args=hyperparameters)
                elif operation == 'test':
                    mgr.test_model(override_args=hyperparameters)
            except Exception as e:
                error_type = type(e).__name__
                print(f'{error_type}: {e}')
