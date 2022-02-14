#!/usr/bin/python3
import os

from typing import Dict
import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import torch
import random
import re

from models.base import Base
from evaluation import Evaluation


class ModelManager:
    @property
    def model_class(self):
        return self.__model_class

    @model_class.setter
    def model_class(self, model_class):
        self.__model_class = model_class

    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset):
        self.__dataset = dataset

    @property
    def tag(self):
        return self.__tag

    @tag.setter
    def tag(self, tag):
        self.__tag = tag

    @property
    def datasets_dir(self):
        return self.__datasets_dir

    @datasets_dir.setter
    def datasets_dir(self, datasets_dir):
        self.__datasets_dir = datasets_dir

    @property
    def results_dir(self):
        return self.__results_dir

    @results_dir.setter
    def results_dir(self, results_dir):
        self.__results_dir = results_dir

    @property
    def model_name(self):
        return self.model_class.__name__

    @property
    def system(self):
        return self.dataset.split('_')[0]

    @property
    def checkpoint_dir(self):
        return os.path.join(self.results_dir, self.tag, self.model_name, self.system, 'checkpoints')

    @property
    def history_dir(self):
        return os.path.join(self.results_dir, self.tag, self.model_name, self.system, 'histories')

    @property
    def plot_dir(self):
        return os.path.join(self.results_dir, self.tag, self.model_name, self.system, 'plots')

    def __init__(self, model_class: type, dataset: str, tag: str = 'essential', quiet=False):
        self.model_class = model_class
        self.dataset = dataset
        self.tag = tag
        self.datasets_dir = 'data'
        self.results_dir = 'results'
        self.quiet = quiet
        self.cond_print('> initializing model manager')

        self.__default_parameters = {'dimensions': 3,
                               'input_steps': 150,
                               'output_steps': 200,
                               'latent_dim': 256,
                               'batch_size': 128,
                               'learning_rate': 1e-3,
                               'gamma': 0.6,
                               'plateau': 10,
                               'loss': 'mse',
                               'epochs': 10000,
                               'tf_factor': 1.00,
                               'curriculum_length': 1000,
                               'patience': 100}

    def cond_print(self, *args, **kwargs):
        if not self.quiet:
            print(*args, **kwargs)

    @staticmethod
    def setting_seeds():
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    @staticmethod
    def denormalize(data, mean, std):
        denormalized_data = data * std + mean
        return denormalized_data

    def train_model(self, override_args: Dict = None, force_new: bool = False):
        self.cond_print('> training model {}'.format(self.model_name))
        override_args = override_args or {}

        # setting hyper parameters
        hyper_parameters = self.__default_parameters.copy()
        for k, v in override_args.items():
            hyper_parameters[k] = v

        # setting seeds
        self.setting_seeds()

        # creating model
        model = self.model_class(hyper_parameters=hyper_parameters, force_new=force_new)

        generators = model.generate_generators(datasets_dir=self.datasets_dir, dataset=self.dataset)
        model.fit_generator(generators, self.checkpoint_dir, self.history_dir)

        return model

    def test_model(self, override_args: Dict = None):
        self.cond_print('> testing model {}'.format(self.model_name))
        override_args = override_args or {}

        # setting hyper parameters
        hyper_parameters = self.__default_parameters.copy()
        for k, v in override_args.items():
            hyper_parameters[k] = v

        # setting seeds
        self.setting_seeds()
        self.cond_print('> building model')
        model = self.model_class(hyper_parameters=hyper_parameters)
        self.cond_print('> loading weights')
        model.load_weights(self.checkpoint_dir)

        self.cond_print('> generating time series data generators')
        generators = model.generate_generators(datasets_dir=self.datasets_dir, dataset=self.dataset)


        self.cond_print('> predicting time series data')
        predictions, realities = model.predict_generator(generators=generators)

        predictions = self.denormalize(predictions, mean=model.generators.mean, std=model.generators.std)
        realities = self.denormalize(realities, mean=model.generators.mean, std=model.generators.std)

        strategy = re.sub(r'([A-Z_]+)(_Lin|_InvSig|Exp)?$', r'\1', self.model_name)

        decay = re.findall('Lin|InvSig|Exp', self.model_name)
        if len(decay) == 0:
            decay = 'Const'
        elif len(decay) == 1:
            decay = decay[0]
        else:
            self.cond_print('decay: {} looks strange'.format(decay))

        self.cond_print('> computing metrics')
        return [generators.system] + self.compute_metrics(model=model, strategy=strategy, curriculum=decay, predictions=predictions, realities=realities)

    def plot_history(self, model: Base, system: str, epoch: int, base_filename: str, save_plot: bool = True, suffix='jpg', fix_scale=False):
        history_filename = 'history.csv'
        history_filepath = os.path.join(self.history_dir, self.model_name, system, base_filename, history_filename)
        plt.clf()
        try:
            # linear y axis
            df = pd.read_csv(history_filepath)[:epoch]
            xlim = [0, model.epochs] if fix_scale else [None, None]
            ylim = [0, 1] if fix_scale else [None, None]
            df['min_loss'] = df['loss'].min()
            df['min_val_loss'] = df['val_loss'].min()
            df.plot(x='epoch', title=self.model_name, ylim=ylim, xlim=xlim)
            if save_plot:
                save_to = 'history.{}'.format(suffix)
                model_plot_dir = os.path.join(self.plot_dir, self.model_name, system, base_filename)
                os.makedirs(model_plot_dir, exist_ok=True)
                model_plot_path = os.path.join(model_plot_dir, save_to)
                plt.ylabel('values', fontsize=18)
                plt.ylabel('epoch', fontsize=18)
                plt.tick_params(axis='x', labelsize=16)
                plt.tick_params(axis='y', labelsize=16)
                #plt.legend(['training', 'validation'])
                plt.savefig(model_plot_path)
                plt.close()
            else:
                plt.show()
            plt.clf()
            # log y axis
            xlim = [0, model.epochs] if fix_scale else [None, None]
            ylim = [1e-7, 1.] if fix_scale else [None, None]
            df.plot(x='epoch', title=self.model_name, logy=True, ylim=ylim, xlim=xlim)
            if save_plot:
                save_to = 'log_history.{}'.format(suffix)
                model_plot_dir = os.path.join(self.plot_dir, self.model_name, system, base_filename)
                os.makedirs(model_plot_dir, exist_ok=True)
                model_plot_path = os.path.join(model_plot_dir, save_to)
                plt.ylabel('values', fontsize=18)
                plt.xlabel('epoch', fontsize=18)
                plt.tick_params(axis='x', labelsize=16)
                #plt.tick_params(axis='y', labelsize=16)
                #plt.legend(['training', 'validation'])
                plt.savefig(model_plot_path)
                plt.close()
            else:
                plt.show()
        except Exception as e:
            print('[ERROR] Could not plot data from {}: "{}"'.format(history_filepath, e))

    def plot_diff(self, prediction, reality, model_name, system, base_filename, save_to=None, loss=None):
        if loss is None:
            loss = lambda x, y: torch.nn.MSELoss(x, y).numpy()

        if len(prediction) != len(reality):
            raise ValueError('prediction and reality must have the same length: {} != {}'.format(len(prediction),
                                                                                                 len(reality)))
        distances = []
        for i in range(len(prediction)):
            distances.append(loss(prediction[i], reality[i]).item())
        plt.plot(distances)
        plt.yscale('log')
        plt.ylim(bottom=1e-10, top=1)
        plt.scatter(x=range(1, len(distances) + 1), y=distances)
        plt.ylabel('distance')
        plt.xlabel('step')
        plt.title(model_name)

        if save_to is None:
            plt.show()
        else:
            plt.title('{} - {}'.format(model_name, save_to))
            diff_plot_dir = os.path.join(self.plot_dir, model_name, system, base_filename)
            os.makedirs(diff_plot_dir, exist_ok=True)
            save_to = 'diff_{}'.format(save_to)
            diff_plot_path = os.path.join(diff_plot_dir, save_to)
            plt.savefig(diff_plot_path)
            plt.close()

    def plot_r2_per_step(self, predictions: torch.Tensor, realities: torch.Tensor, model: Base, system, base_filename, fix_scale=True, save_to=None):
        model_name = model.__class__.__name__

        if len(predictions) != len(realities):
            raise ValueError('prediction and reality must have the same length: {} != {}'.format(len(predictions),
                                                                                                 len(realities)))

        mean_r2_scores, _ = Evaluation.r2_scores(predictions, realities, model.generators.step_width, model.generators.lle)
        plt.clf()
        plt.plot(mean_r2_scores, label='mean')
        #print(model.__class__.__name__, *mean_r2_scores, sep=',')
        plt.plot([0.7 for _ in range(len(mean_r2_scores))], label='limit')
        plt.ylabel('R2-Score')
        plt.xlabel('step')
        if fix_scale:
            plt.ylim(bottom=0, top=1.1)
            plt.xlim(left=0, right=model.output_steps)
        plt.legend(fontsize='small')
        plt.title('{} - Mean distance ({} samples)'.format(model_name, len(predictions)))

        if save_to is None:
            plt.show()
        else:
            diff_plot_dir = os.path.join(self.plot_dir, model_name, system, base_filename)
            os.makedirs(diff_plot_dir, exist_ok=True)
            save_to = 'r2_per_step_{}'.format(save_to)
            diff_plot_path = os.path.join(diff_plot_dir, save_to)
            plt.savefig(diff_plot_path)
            plt.close()

    def plot_rmse_per_step(self, predictions: torch.Tensor, realities: torch.Tensor, model: Base, system,
                           base_filename, fix_scale=True, save_to=None):
        model_name = model.__class__.__name__

        if len(predictions) != len(realities):
            raise ValueError('prediction and reality must have the same length: {} != {}'.format(len(predictions),
                                                                                                 len(realities)))

        mean_rmses = Evaluation.rmses(predictions, realities)
        plt.clf()
        plt.plot(mean_rmses, label='mean')
        plt.ylabel('RMSE')
        plt.xlabel('step')
        if fix_scale:
            plt.ylim(bottom=0, top=6.0)
            plt.xlim(left=0, right=model.output_steps)
        plt.legend(fontsize='small')
        plt.title('{} - Mean distance ({} samples)'.format(model_name, len(predictions)))

        if save_to is None:
            plt.show()
        else:
            diff_plot_dir = os.path.join(self.plot_dir, model_name, system, base_filename)
            os.makedirs(diff_plot_dir, exist_ok=True)
            save_to = 'rmse_per_step_{}'.format(save_to)
            diff_plot_path = os.path.join(diff_plot_dir, save_to)
            plt.savefig(diff_plot_path)
            plt.close()

    def compute_metrics(self, model: Base, strategy: str, curriculum: str, predictions: np.ndarray, realities: np.ndarray):
        nrmse = Evaluation.median_nrmse(torch.tensor(predictions), torch.tensor(realities), model.generators.std.mean())

        to_steps = max(model.output_steps // 10, 1)
        last_10_nrmse = Evaluation.m_nrmse_of_steps(predictions=torch.tensor(predictions), targets=torch.tensor(realities), norm_value=model.generators.std.mean(), from_step=-to_steps)
        r2s, lt_r2_09 = Evaluation.median_r2_scores(torch.tensor(predictions), torch.tensor(realities), step_width=model.generators.step_width, lle=model.generators.lle, limit=0.9)
        r2 = r2s[-1]
        return [strategy, curriculum, round(nrmse, 6), round(r2, 6), round(last_10_nrmse, 6)]

    def load_latest_checkpoint(self, tag: str):
        max_epoch = 0
        matching_checkpoint = None
        for checkpoint in os.listdir(self.checkpoint_dir):
            _, found_tag, found_epoch, suffix = re.split(r'[-.]', checkpoint)
            if tag != found_tag:
                continue
            if int(found_epoch) > max_epoch:
                matching_checkpoint = checkpoint
                max_epoch = int(found_epoch)

        return matching_checkpoint, max_epoch

    def results_plot(self, file_path):
        df = pd.read_csv(os.path.join(self.results_dir, file_path), usecols=['strategy', 'decay', 'epsilon', 'rmse', 'm_rmse', 'nrmse', 'm_nrmse', 'r2'])
        figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
        system = file_path.split('_')[0]

        metrics = ['rmse', 'm_rmse', 'nrmse', 'm_nrmse']
        metrics = ['nrmse']
        for metric in metrics:
            min_rmse = df.groupby(by='strategy').min()[metric]
            min_rmse = min_rmse.sort_values(ascending=False)
            a = min_rmse.plot.bar(rot=0)
            plt.xticks(fontsize=8)
            plt.title(metric)
            plt.yscale('log')
            plt.savefig(os.path.join(self.plot_dir, '{}_min_{}_plot.pdf'.format(system, metric)))
            plt.clf()

            bp_df = df[['strategy', 'nrmse']]
            bp_df.plot()
            plt.savefig(os.path.join(self.plot_dir, '{}_boxplot_{}.pdf'.format(system, metric)))

            #df[metric].plot.bar()
            #plt.yscale('log')
            #plt.savefig(os.path.join(self.plot_dir, '{}_grouped_{}.pdf'.format(system, metric)))
            
            #plt.show()
