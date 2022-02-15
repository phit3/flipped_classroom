#!/usr/bin/python3
import argparse
import re
import json
from prettytable import PrettyTable

from model_manager import ModelManager
from models.tf import TF
from models.fr import FR
from models.cl_ctf_p import CL_CTF_P
from models.cl_dtf import CL_DTF_P_Lin, CL_DTF_P_InvSig, CL_DTF_P_Exp, CL_DTF_D_Lin, CL_DTF_D_InvSig, CL_DTF_D_Exp
from models.cl_itf import CL_ITF_P_Lin, CL_ITF_P_InvSig, CL_ITF_P_Exp, CL_ITF_D_Lin, CL_ITF_D_InvSig, CL_ITF_D_Exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replication package for: Flipped Classroom: Effective Teaching for Chaotic Time Series Forecasting.')
    parser.add_argument('-m', '--models', dest='model_class_name', nargs='+',
                        help='class name(s) of the model(s) to train or test.')
    parser.add_argument('-o', '--operation', dest='operation', default='test', choices=['train', 'test'], help='operation that shall be performed.')
    parser.add_argument('-d', '--datasets', dest='dataset', nargs='+',
                        help='datasets to be processed optionally together with the desired hyperparameters in JSON format (e.g. \'{"thomas_0.1_0.055": {"gamma": 0.6, "lr": 1e-3, "plateau": 10, "latent_dim": 256, "output_steps": 182}}\').')
    parser.add_argument('-t', '--tag', dest='tag', type=str, default='test', help='tag that identifies a set of experiments')
    parser.add_argument('-s', '--skip-missing', dest='skip_missing', default=False, action='store_true', help='skips model-dataset combination tests for which no checkpoint is provided for the given tag')
    parser.add_argument('-q', '--quiet', dest='quiet', default=False, action='store_true', help='reduces noise on your command line.')

    args = parser.parse_args()
    operation = args.operation
    tag = args.tag
    skip_missing = args.skip_missing
    quiet = args.quiet

    hyperparameter_sets = {'hyperroessler':
                               {'filename': 'hyperroessler_0.1_0.14',
                                'dimensions': 4,
                                'lr': 3e-5,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 72},
                           'lorenz96':
                               {'filename': 'lorenz96_0.05_1.67',
                                'dimensions': 40,
                                'lr': 1e-3,
                                'gamma': 0.9,
                                'plateau': 20,
                                'latent_dim': 256,
                                'output_steps': 12},
                           'lorenz':
                               {'filename': 'lorenz_0.01_0.905',
                                'dimensions': 3,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 111},
                           'mackeyglass':
                               {'filename': 'mackeyglass_1.0_0.006',
                                'dimensions': 1,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 167},
                           'roessler':
                               {'filename': 'roessler_0.12_0.069',
                                'dimensions': 3,
                                'lr': 1e-3,
                                'gamma': 0.6,
                                'plateau': 10,
                                'latent_dim': 256,
                                'output_steps': 121},
                           'thomas':
                               {'filename': 'thomas_0.1_0.055',
                                'dimensions': 3,
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

    metrics = PrettyTable()
    metrics.field_names = ['Dataset', 'Strategy', 'Curriculum', 'NRMSE', 'R2', 'Last 10% NRMSE']
    metrics.align['Dataset'] = 'l'
    metrics.align['Strategy'] = 'l'
    metrics.align['Curriculum'] = 'l'
    metrics.align['NRMSE'] = 'r'
    metrics.align['R2'] = 'r'
    metrics.align['Last 10% NRMSE'] = 'r'
    for dataset, override_args in datasets.items():
        dataset = re.sub(r'.csv$', '', dataset)
        if not override_args:
            override_args = {}
        if dataset in hyperparameter_sets:
            hyperparameters = hyperparameter_sets[dataset]
            filename = hyperparameters['filename']
        else:
            print(f'[WARNING] Did not find any predefined hyperparameters for {dataset} - using default set as basis.')
            hyperparameters = hyperparameter_sets['default']
            filename = dataset

        for hp, new in override_args.items():
            if hp in hyperparameters:
                old = hyperparameters[hp]
                if not quiet:
                    print(f'Overriding {hp}: {old} -> {new}.')
            hyperparameters[hp] = new

        for model_name in args.model_class_name:
            #if os.path.isfile(os.path.join()
            try:
                model_class = eval(model_name)
                print('Running {} {} operation for {} on {}.'. format(tag, operation, model_name, dataset))
                mgr = ModelManager(model_class=model_class, dataset=filename, tag=tag, quiet=quiet)
                if operation == 'train':
                    mgr.train_model(override_args=hyperparameters)
                elif operation == 'test':
                    try:
                        row = mgr.test_model(override_args=hyperparameters)
                        metrics.add_row(row=row)
                    except FileNotFoundError as e:
                        if skip_missing:
                            if not quiet:
                                print('> skipping')
                            continue
                        else:
                            raise e
            except Exception as e:
                error_type = type(e).__name__
                print(f'{error_type}: {e}')
    print(metrics)
