#!/usr/bin/python3
import argparse
import re

from model_manager import ModelManager
from models.tf import TF
from models.fr import FR
from models.cl_ctf_p import CL_CTF_P
from models.cl_dtf import CL_DTF_P_Lin, CL_DTF_P_InvSig, CL_DTF_P_Exp, CL_DTF_D_Lin, CL_DTF_D_InvSig, CL_DTF_D_Exp
from models.cl_itf import CL_ITF_P_Lin, CL_ITF_P_InvSig, CL_ITF_P_Exp, CL_ITF_D_Lin, CL_ITF_D_InvSig, CL_ITF_D_Exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Replication package for: Flipped Classroom: Effective Teaching for Chaotic Time Series Forecasting.')
    parser.add_argument('--models', dest='model_names', action='extend', nargs='+', default=[],
                        choices=['TF', 'FR', 'CL_CTF_P',
                                 'CL_DTF_P_Lin', 'CL_DTF_P_InvSig', 'CL_DTF_P_Exp',
                                 'CL_DTF_D_Lin', 'CL_DTF_D_InvSig', 'CL_DTF_D_Exp',
                                 'CL_ITF_P_Lin', 'CL_ITF_P_InvSig', 'CL_ITF_P_Exp',
                                 'CL_ITF_D_Lin', 'CL_ITF_D_InvSig', 'CL_ITF_D_Exp'], help='Model to use.')
    parser.add_argument('--dataset', dest='dataset', help='Dataset file to be processed.')
    parser.add_argument('--operation', dest='operation', default='test', choices=['train', 'test'], help='Operation that shall be performed.')
    parser.add_argument('--lr', dest='lr', type=float, default=None, help='Initial learning rate used for training.')
    parser.add_argument('--gamma', dest='gamma', type=float, default=None, help='Factor to be applied to the learning rate by the learning rate scheduler.')
    parser.add_argument('--plateau', dest='plateau', type=int, default=None, help='Plateau or patience used by the learning rate scheduler.')
    parser.add_argument('--latent-dim', dest='latent_dim', type=int, default=None, help='Latent dimension or state size of the GRUs.')
    parser.add_argument('--output-steps', dest='output_steps', type=int, default=None, help='Latent dimension or state size of the GRUs.')
    parser.add_argument('--tag', dest='tag', type=str, default='test', help='Tag that denotes the set of experiments')
    parser.add_argument('--quiet', dest='quiet', default=False, action='store_true', help='Providing less out prints.')

    args = parser.parse_args()

    dataset = re.sub(r'.csv$', '', args.dataset)
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
                               {'dimensions': 3,
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
                           }

    override_args = hyperparameter_sets[dataset]

    if args.lr:
        print('Overriding lr')
        override_args['lr'] = args.lr
    if args.gamma:
        print('Overriding gamma')
        override_args['gamma'] = args.gamma
    if args.plateau:
        print('Overriding plateau')
        override_args['plateau'] = args.gamma
    if args.latent_dim:
        print('Overriding latent_dim')
        override_args['latent_dim'] = args.latent_dim
    if args.output_steps:
        print('Overriding output_steps')
        override_args['output_steps'] = args.output_steps

    for model_name in args.model_names:
        model_class = eval(model_name)
        if quiet:
            print('Running {} {} operation for {} on {}.'. format(tag, operation, model_name, dataset))
        mgr = ModelManager(model_class=model_class, dataset=dataset, tag=tag, quiet=quiet)
        if operation == 'train':
            mgr.train_model(override_args=override_args)
        elif operation == 'test':
            mgr.test_model(override_args=override_args)
