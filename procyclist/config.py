import abc

import numpy as np
import tensorflow as tf


class Config(metaclass=abc.ABCMeta):
    _attributes = [
        'batch_size',
        'dropout',
        'input_dim',
        'inputs',
        'learning_rate',
        'max_time',
        'n_epochs',
        'n_hidden',
        'n_layers',
        'output_dim',
        'outputs',
    ]

    @classmethod
    @abc.abstractmethod
    def load(cls, infile):
        return np.load(infile)

    def save(self, outfile):
        archive = dict((attr, getattr(self, attr)) for attr in self._attributes)
        return np.savez_compressed(outfile, **archive)


class DefaultConfig(Config):
    # Training parameters
    batch_size = 500
    learning_rate = 0.01
    max_time = 5000
    n_epochs = 2000

    # Model parameters
    dropout = 0.7
    n_hidden = 100
    n_layers = 2

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.input_dim = len(inputs)
        self.output_dim = len(outputs)

        # Initialize optimizer
        optimizer = tf.train.AdamOptimizer
        self.optimizer = optimizer(learning_rate=self.learning_rate)

    @classmethod
    def load(cls, infile):
        archive = super(cls).load(infile)
        self = cls(archive['inputs'], archive['outputs'])
        for attr in self._attributes:
            setattr(self, attr, archive[attr])
        return self
