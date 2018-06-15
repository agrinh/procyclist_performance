import datetime

import numpy as np
import tensorflow as tf


class SequenceBuffer:

    def __init__(self, size, sequence, dtype=np.object):
        self.__buffer = np.zeros((size, ), dtype=dtype)
        self.__filled = np.full((size, ), False, dtype=np.bool)
        self.__sequence = sequence
        self.__refill()

    @property
    def content(self):
        return self.__buffer

    def evict(self, selection):
        self.__filled[selection] = False
        return self.__refill()

    def __refill(self):
        indices = np.where(~self.__filled)[0]
        for location, item in zip(indices, self.__sequence):
            self.__buffer[location] = item
            self.__filled[location] = True

        # resize buffer if sequence is drained
        filled = self.__filled
        self.__buffer = self.__buffer[filled]
        self.__filled = self.__filled[filled]
        return filled

    def __len__(self):
        return len(self.__buffer)


def time_string():
    time_str = str(datetime.datetime.now())
    return time_str.replace(' ', '_').replace(':', '_').replace('.', '_')


def clear_lstm_state(state, drop_mask, remove=0):
    """
    Clears lstm state according to drop_mask

    Removes rows correpsponding to the 'remove' last True values in drop_mask

    Parameters
    ----------
    state : tuple of LSTMStateTuple
        State with n rows (samples)
    drop_mask : numpy.array(dtype=numpy.bool)
        Boolean mask specifying which rows in the state should be cleared
    remove : int
        The number of rows set for clearing that should be removed.

    Returns
    -------
    tuple of LSTMStateTuple
        New state with rows cleared and removed according to parameters
    """
    # Determine rows to be removed
    dropped = np.where(drop_mask)[0]
    dropped = dropped[(len(dropped) - remove):]

    # Construct new state
    new_state = list()
    for s in state:
        s.c[drop_mask] = 0
        s.h[drop_mask] = 0
        new_state.append(
            tf.nn.rnn_cell.LSTMStateTuple(
                np.delete(s.c, dropped, axis=0),
                np.delete(s.h, dropped, axis=0)
            )
        )
    return tuple(new_state)


def resample(session, factor):
    resampled = np.arange(0, len(session), 1/factor)
    original = np.arange(len(session))
    resample = lambda y: np.interp(resampled, original, y)
    return np.apply_along_axis(resample, 0, session)


def simple_summary(values):
    return tf.Summary(value=[
        tf.Summary.Value(tag=name, simple_value=value)
        for name, value in values.items()
    ])
