import configparser
import glob
import inspect
import hashlib
import os

import numpy as np

from procyclist.sessions import Sessions
from procyclist.utilities import resample


def load():
    # Use hash of this function to determine if dataset should be recreated
    current_source = inspect.getsource(load)
    current_hash = hashlib.sha256(current_source.encode()).hexdigest()
    config = configparser.ConfigParser()
    config.read(Sessions.SESSIONS_CONFIG)
    cache_path = config['DEFAULT']['cache_path']
    cache_file = os.path.join(cache_path, current_hash + '.npz')

    # Make sure path exists. Load from dataset cache if exists, otherwise clear
    # all old cache files.
    if cache_path:
        if os.path.isdir(cache_path):
            if os.path.isfile(cache_file):
                return Sessions.load(cache_file)
            else:
                for old_cache in glob.glob(os.path.join(cache_path, '*.npz')):
                    os.remove(old_cache)
        else:
            os.makedirs(cache_path)


    # Create dataset from files on disk with desired parameters
    parameters = [
        'Time',
        'Speed',
        'Distance',
        'Power',
        'Cadence',
        'power/kg',
        'Altitude',
        'Heart Rate'
    ]
    sessions = Sessions.create(
        name='men',
        device='srm',
        parameters=parameters
    )

    # Resample all sequences
    sessions.transform(lambda x: resample(x, 0.3))

    # Remove sessions where heart rate monitor obviously not attached, either
    # on a very low mean value or a large fraction being zero.
    i_hr = sessions.parameters.index('Heart Rate')
    sessions = sessions[sessions.map(lambda x: x[:, i_hr].mean()) > 20]
    sessions = sessions[sessions.map(lambda x: np.sum(x[:, i_hr] == 0)
                                               < 0.05 * len(x))]

    # Remove sessions where heart rate is unrealistically high. Allow for some
    # noise depending on how high the heart rate is.
    over_210 = sessions.map(lambda x: np.sum(x[:, i_hr] > 210))
    over_240 = sessions.map(lambda x: np.sum(x[:, i_hr] > 240))
    sessions = sessions[np.logical_and(over_210 < 50, over_240 < 5)]

    # Remove sessions with negative power, speed or distance
    positive_parameters = [
        'Distance',
        'Power',
        'Speed',
    ]
    i_positive = list(map(sessions.parameters.index, positive_parameters))
    sessions = sessions[sessions.map(lambda x: np.all(x[:, i_positive] >= 0))]

    # Remove sessions where distance is not increasing while power is recorded
    # for a large number of timesteps.
    i_power = sessions.parameters.index('Power')
    i_distance = sessions.parameters.index('Distance')
    missing_distance = sessions.map(
        lambda x: np.sum(
            np.logical_and(np.diff(x[:, i_distance]) == 0, x[1:, i_power] > 0))
    )
    sessions = sessions[missing_distance < sessions.meta.length * 0.1]

    # Trim sessions to remove leading and trailing zeros in heart rate
    sessions.transform(lambda x: x[(x[:, -1] > 0).argmax():])
    sessions.transform(lambda x: x[:len(x)-(x[::-1, -1] > 0).argmax()])

    # Compute means and standard deviations
    sums = sessions.map(lambda x: np.sum(x, axis=0))
    means = np.sum(sums, axis=0) / np.sum(sessions.meta.length)
    squares = sessions.map(lambda x: np.sum((x - means)**2, axis=0))
    std = np.sqrt(np.sum(squares  / np.sum(sessions.meta.length), axis=0))
    std[std == 0] = 1

    # Normalize all w.r.t. mean and standard deviation except heart rate
    std[-1] = 1
    means[-1] = 0
    sessions.transform(lambda x: (x - means) / std)
    sessions.global_meta['means'] = means
    sessions.global_meta['std'] = std

    ## Derive time deltas
    #sessions.derive(
    #    'Time delta',
    #    lambda x: np.concatenate(
    #        [0], np.diff(x[:, 1]) / np.maximum(x[1:, 0], [1e-6])
    #    )
    #)

    # Derive delayed heart rate
    delay = 30
    sessions.derive(
        'Delayed Heart Rate',
        lambda x:  np.pad(
            x[:-delay, i_hr], [min(delay, len(x)), 0], mode='constant')
    )

    # Store session in cache if path available
    if cache_path:
        sessions.save(cache_file)
    return sessions
