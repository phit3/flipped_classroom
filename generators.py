import pandas as pd
import numpy as np
import os
import random


class SeqGenerators:
    @property
    def dataset(self):
        return self.__dataset

    @dataset.setter
    def dataset(self, dataset):
        self.__dataset = dataset

    @property
    def system(self):
        return self.dataset.split('_')[0]

    @property
    def step_width(self):
        return float(self.dataset.split('_')[1])

    @property
    def lle(self):
        return float(self.dataset.split('_')[2])

    @property
    def input_steps(self):
        return self.__input_steps

    @input_steps.setter
    def input_steps(self, input_steps):
        self.__input_steps = input_steps

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size):
        self.__batch_size = batch_size

    @property
    def dimensions(self):
        return self.__dimensions

    @dimensions.setter
    def dimensions(self, dimensions):
        self.__dimensions = dimensions

    @property
    def data(self):
        return self.__data
    
    @data.setter
    def data(self, data):
        self.__data = data

    @property
    def current_idx(self):
        return self.__current_idx

    @current_idx.setter
    def current_idx(self, current_idx):
        self.__current_idx = current_idx

    @property
    def output_steps(self):
        return self.__output_steps

    @output_steps.setter
    def output_steps(self, output_steps):
        self.__output_steps = output_steps

    @property
    def samples(self):
        return self.__samples
    
    @samples.setter
    def samples(self, samples):
        self.__samples = samples

    @property
    def train_samples(self):
        return int(self.train_split * self.samples)

    @property
    def valid_samples(self):
        return int(self.valid_split * self.samples)

    @property
    def test_samples(self):
        return int(self.test_split * self.samples)

    @property
    def train_split(self):
        return self.__train_split

    @train_split.setter
    def train_split(self, train_split):
        self.__train_split = train_split

    @property
    def valid_split(self):
        return self.__valid_split

    @valid_split.setter
    def valid_split(self, valid_split):
        self.__valid_split = valid_split

    @property
    def test_split(self):
        return self.__test_split

    @test_split.setter
    def test_split(self, test_split):
        self.__test_split = test_split

    @property
    def current_idx_idx(self):
        return self.__current_idx_idx

    @current_idx_idx.setter
    def current_idx_idx(self, current_idx_idx):
        self.__current_idx_idx = current_idx_idx

    @property
    def index_mapper(self):
        return self.__index_mapper

    @index_mapper.setter
    def index_mapper(self, index_mapper):
        self.__index_mapper = index_mapper

    @property
    def train_index_mapper(self):
        return self.index_mapper[:self.train_samples]

    @property
    def valid_index_mapper(self):
        return self.index_mapper[self.train_samples:self.train_samples + self.valid_samples]

    @property
    def test_index_mapper(self):
        return self.index_mapper[-self.test_samples:]

    @property
    def mean(self) -> float:
        return self.__mean

    @mean.setter
    def mean(self, mean: float) -> None:
        self.__mean = mean

    @property
    def std(self) -> float:
        return self.__std

    @std.setter
    def std(self, std: float) -> None:
        self.__std = std

    def __init__(self, datasets_dir: str, dataset: str, input_steps: int, batch_size: int, max_samples: int = 10000):
        self.dataset = dataset
        self.data_file_location = os.path.join(datasets_dir, '{}.csv'.format(self.dataset))
        all_data = np.array(pd.read_csv(self.data_file_location))
        self.input_steps = input_steps
        self.batch_size = batch_size
        self.dimensions = all_data.shape[1]
        self.data = all_data
        self.mean = self.data.mean(axis=0)
        self.std = self.data.std(axis=0)
        self.data = (self.data - self.mean) / (self.std + 1e-18)
        self.output_steps = input_steps

        self.train_split, self.valid_split, self.test_split = 0.8, 0.1, 0.1

        self._prepare_index_mapper(max_samples)

    def _prepare_index_mapper(self, max_samples):
        different_samples = (len(self.data) - (self.output_steps + self.input_steps - 1)) // (self.output_steps + self.input_steps)
        if max_samples is None or max_samples > different_samples:
            self.samples = different_samples
        else:
            self.samples = max_samples
        self.current_idx_idx = 0
        self.index_mapper = list(range(0, len(self.data) - (self.output_steps + self.input_steps - 1),
                                       self.input_steps + self.output_steps))
        random.shuffle(self.index_mapper)
        if len(self.index_mapper) < self.samples:
            print('[WARN] Could only generate {} samples instead of desired {} samples'.format(len(self.index_mapper),
                                                                                               self.samples))
        self.index_mapper = self.index_mapper[:self.samples]

        self.current_idx = self.index_mapper[self.current_idx_idx]

    def generate_train(self):
        return self.generate(self.train_index_mapper)

    def generate_valid(self):
        return self.generate(self.valid_index_mapper)

    def generate_test(self):
        return self.generate(self.test_index_mapper)

    def generate(self, index_mapper):
        raise NotImplementedError('Default generate() is not implemented! Please inherit from Generators.')

    def __len__(self):
        return self.samples


class Seq2SeqGenerators(SeqGenerators):
    def __init__(self, datasets_dir: str, dataset: str, input_steps: int, output_steps: int,
                 batch_size: int, max_samples: int = 10000):
        super(Seq2SeqGenerators, self).__init__(datasets_dir=datasets_dir, dataset=dataset,
                                                input_steps=input_steps, batch_size=batch_size,
                                                max_samples=max_samples)
        self.output_steps = output_steps
        self._prepare_index_mapper(max_samples)

    def generate(self, index_mapper):
        x_encoder = np.zeros((self.batch_size, self.input_steps, self.dimensions))
        x_decoder = np.zeros((self.batch_size, self.output_steps, self.dimensions))
        y = np.zeros((self.batch_size, self.output_steps, self.dimensions))
        while True:
            for i in range(self.batch_size):
                if self.current_idx_idx >= len(index_mapper):
                    self.current_idx_idx = 0
                self.current_idx = index_mapper[self.current_idx_idx]
                x_encoder[i, :, :] = self.data[self.current_idx: self.current_idx + self.input_steps]
                x_decoder[i, :, :] = self.data[self.current_idx + self.input_steps - 1:
                                               self.current_idx + self.input_steps + self.output_steps - 1]
                y[i, :, :] = self.data[self.current_idx + self.input_steps:
                                       self.current_idx + self.input_steps + self.output_steps]
                self.current_idx_idx = self.current_idx_idx + 1
            batch = [x_encoder, x_decoder], y
            yield batch

    def data_samples(self, offset=0, count=5):
        if offset + count + self.input_steps + self.output_steps > self.test_samples:
            raise ValueError('Given count value {} is too large. It may be {} at most for an offset of {}.'.format(count, self.samples - offset - self.output_steps - self.input_steps, offset))
        data_samples = np.zeros((count, self.input_steps, self.dimensions))
        realities = np.zeros((count, self.output_steps, self.dimensions))
        random_range = list(range(count))
        random.shuffle(random_range)
        for i in random_range:
            data_samples[i, :, :] = self.data[self.train_samples + self.valid_samples + offset + i:  self.train_samples + self.valid_samples + offset + i + self.input_steps]
            realities[i, :, :] = self.data[self.train_samples + self.valid_samples + offset + i + self.input_steps: self.train_samples + self.valid_samples + offset + i + self.input_steps + self.output_steps]
        return data_samples, realities
