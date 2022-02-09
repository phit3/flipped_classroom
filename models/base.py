import os
import sys
import logging
import typing
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from generators import SeqGenerators


class Base:
    class Model(nn.Module):
        def __init__(self):
            super(Base.Model, self).__init__()

        def forward(self):
            pass

        def _forward_unimplemented(self, *input: Any) -> None:
            pass

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    @property
    def model_name(self):
        return self.__class__.__name__

    @property
    def dimensions(self):
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dimensions: int):
        self.__dimensions = dimensions

    @property
    def input_steps(self):
        return self.__input_steps

    @input_steps.setter
    def input_steps(self, input_steps):
        self.__input_steps = input_steps

    @property
    def output_steps(self):
        return self.__output_steps

    @output_steps.setter
    def output_steps(self, output_steps):
        self.__output_steps = output_steps

    @property
    def latent_dim(self):
        return self.__latent_dim

    @latent_dim.setter
    def latent_dim(self, latent_dim):
        self.__latent_dim = latent_dim

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.__batch_size = batch_size

    @property
    def learning_rate(self):
        return self.__learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.__learning_rate = learning_rate

    @property
    def gamma(self):
        return self.__gamma

    @gamma.setter
    def gamma(self, gamma):
        self.__gamma = gamma

    @property
    def plateau(self):
        return self.__plateau

    @plateau.setter
    def plateau(self, plateau):
        self.__plateau = plateau

    @property
    def tf_ratio(self):
        return self.__tf_ratio

    @tf_ratio.setter
    def tf_ratio(self, tf_ratio):
        self.__tf_ratio = tf_ratio

    @property
    def curriculum_length(self):
        return self.__curriculum_length

    @curriculum_length.setter
    def curriculum_length(self, curriculum_length):
        self.__curriculum_length = curriculum_length

    @property
    def loss(self):
        return self.__loss

    @loss.setter
    def loss(self, loss):
        self.__loss = loss

    @property
    def loss_fct(self):
        return self.__loss_fct

    @loss_fct.setter
    def loss_fct(self, loss_fct):
        self.__loss_fct = loss_fct

    @property
    def epochs(self):
        return self.__epochs

    @epochs.setter
    def epochs(self, epochs):
        self.__epochs = epochs

    @property
    def generators(self):
        return self.__generators

    @generators.setter
    def generators(self, generators):
        self.__generators = generators

    @property
    def patience(self):
        return self.__patience

    @patience.setter
    def patience(self, patience):
        self.__patience = patience

    @property
    def max_patience(self):
        return self.__max_patience

    @max_patience.setter
    def max_patience(self, max_es_patience):
        self.__max_patience = max(1, max_es_patience)

    @property
    def best_val_loss(self):
        if len(self.__epoch_val_losses) > 1:
            return min(self.__epoch_val_losses[:-1])
        return sys.maxsize

    @property
    def best_loss(self):
        if len(self.__epoch_losses) > 1:
            return min(self.__epoch_losses[:-1])
        return sys.maxsize

    @property
    def last_early_stopping_val_loss(self):
        return self.__last_early_stopping_val_loss

    @last_early_stopping_val_loss.setter
    def last_early_stopping_val_loss(self, last_early_stopping_val_loss):
        self.__last_early_stopping_val_loss = last_early_stopping_val_loss

    @property
    def epoch_loss(self):
        if len(self.__epoch_losses) > 0:
            return self.__epoch_losses[-1]
        return None

    @epoch_loss.setter
    def epoch_loss(self, epoch_loss):
        self.__epoch_losses.append(epoch_loss)

    @property
    def epoch_val_loss(self):
        if len(self.__epoch_val_losses) > 0:
            return self.__epoch_val_losses[-1]
        return None

    @epoch_val_loss.setter
    def epoch_val_loss(self, epoch_val_loss):
        self.__epoch_val_losses.append(epoch_val_loss)

    def get_best_epoch(self, until=None):
        return np.argmin(self.epoch_val_losses[:until])

    @property
    def epoch(self):
        return self.__epoch

    @epoch.setter
    def epoch(self, epoch):
        self.__epoch = epoch

    @property
    def epoch_losses(self):
        return self.__epoch_losses

    @epoch_losses.setter
    def epoch_losses(self, epoch_losses):
        self.__epoch_losses = epoch_losses

    @property
    def epoch_val_losses(self):
        return self.__epoch_val_losses

    @epoch_val_losses.setter
    def epoch_val_losses(self, epoch_val_losses):
        self.__epoch_val_losses = epoch_val_losses

    def loss_of_epoch(self, epoch: int):
        return self.epoch_losses[epoch]

    def val_loss_of_epoch(self, epoch: int):
        return self.epoch_val_losses[epoch]

    @property
    def force_new(self):
        return self.__force_new

    @force_new.setter
    def force_new(self, force_new):
        self.__force_new = force_new

    @property
    def history_started(self):
        return self.__history_started

    @history_started.setter
    def history_started(self, history_started):
        self.__history_started = history_started

    def __init__(self, hyper_parameters: typing.Dict[str, typing.Any], force_new: bool = False):
        super(Base, self).__init__()
        if torch.cuda.device_count() > 0:
            torch.set_default_tensor_type('torch.cuda.DoubleTensor')
        else:
            torch.set_default_tensor_type('torch.DoubleTensor')

        # hyper parameters
        self.dimensions = hyper_parameters.setdefault('dimensions', None)
        self.input_steps = hyper_parameters.setdefault('input_steps', None)
        self.output_steps = hyper_parameters.setdefault('output_steps', None)
        self.latent_dim = hyper_parameters.setdefault('latent_dim', None)
        self.learning_rate = hyper_parameters.setdefault('learning_rate', None)
        self.gamma = hyper_parameters.setdefault('gamma', None)
        self.plateau = hyper_parameters.setdefault('plateau', None)
        self.tf_factor = hyper_parameters.setdefault('tf_factor', None)
        self.curriculum_length = hyper_parameters.setdefault('curriculum_length', None)
        self.loss = hyper_parameters.setdefault('loss', None)
        self.epochs = hyper_parameters.setdefault('epochs', None)
        self.batch_size = hyper_parameters.setdefault('batch_size', None)
        self.patience = hyper_parameters.setdefault('patience', None)
        self.max_patience = self.patience

        self.generators = None
        self.epoch = 0
        self.last_early_stopping_val_loss = None
        self.model = self._build_model()
        self.epoch_val_losses = []
        self.epoch_losses = []
        self.force_new = force_new
        self.history_started = False

        if self.loss == 'mse':
            self.loss_fct = nn.MSELoss()

    def _build_model(self):
        return self.Model()

    def generate_generators(self, datasets_dir: str, dataset: str, max_samples: typing.Optional[int] = None, **kwargs):
        logging.info('Generating generators')
        generators = SeqGenerators(datasets_dir=datasets_dir, dataset=dataset,
                                   input_steps=self.input_steps, batch_size=self.batch_size)
        return generators

    def _build_checkpoint(self, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        for f in os.listdir(checkpoint_dir):
            f_path = os.path.join(checkpoint_dir, f)
            if os.path.isfile(f_path):
                if self.force_new:
                    os.remove(f_path)
                else:
                    raise FileExistsError('The checkpoint {} already exists. Please delete the checkpoint file if '
                                          'you want to retrain the network.'.format(checkpoint_dir))
        return checkpoint_dir

    def save_weights(self, weights_file_path: str):
        torch.save(self.model.state_dict(), weights_file_path)

    def load_weights(self, weights_file_path: str):
        self.model.load_state_dict(torch.load(weights_file_path))

    def write_history(self, history_dir: str, epoch_values):
        if epoch_values is None:
            raise ValueError('epoch_values may not be None at this point')

        os.makedirs(history_dir, exist_ok=True)
        history_file_path = os.path.join(history_dir, 'history.csv')
        if not self.history_started:
            with open(history_file_path, 'w') as f:
                head_line = ','.join(list(epoch_values.keys()))
                f.write('{}\n'.format(head_line))
            self.history_started = True
        with open(history_file_path, 'a+') as f:
            values_line = ','.join(list(epoch_values.values()))
            f.write('{}\n'.format(values_line))

    def early_stopping(self):
        if self.last_early_stopping_val_loss is None or (0.99 * self.last_early_stopping_val_loss) > self.epoch_val_loss:
            self.last_early_stopping_val_loss = self.epoch_val_loss
            self.patience = self.max_patience
        else:
            self.patience -= 1
        return self.patience <= 0

    def print_progress(self, epoch, batch_number, max_batch_number):
        max_length = 40
        bar = '=' * int((max_length * batch_number / max_batch_number))
        space = '.' * (max_length - len(bar) - 1)
        if batch_number < max_batch_number:
            progress_bar = f'\rEpoch {epoch}/{self.epochs} [{bar}>{space}]'
        else:
            progress_bar = f'\rEpoch {epoch}/{self.epochs} [{bar}]'
        print(progress_bar, end='', flush=True)

    def fit_generator(self, generators: SeqGenerators, checkpoint_dir: str, histories_dir: str):
        logging.info('Starting training')
        self.generators = generators
        self._build_checkpoint(checkpoint_dir)

    def predict_generator(self, generators: SeqGenerators):
        self.generators = generators
