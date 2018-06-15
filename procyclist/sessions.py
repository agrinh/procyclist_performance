import configparser
import glob
import os
import pkg_resources
import random
import re

import numpy as np
import pandas as pd
import scipy.io as sio


class Sessions:
    """
    Maintains a dataset of sessions

    Parameters
    ----------
    data : numpy.array(dtype=object)
        Array of numpy matrices. Each matrix is of shape (steps, parameters).
    meta : pandas.DataFrame
        DataFrame of metadata with indices matching the data matrix
    parameters : list of str
        List of parameters with indices matching the corresponding axis of data
    all_parameters : list of str or None
        List of all available parameters if more than those selected
    """

    SESSIONS_CONFIG = pkg_resources.resource_filename(
        'procyclist', 'config/sessions.cfg')
    DEVICES_CONFIG = pkg_resources.resource_filename(
        'procyclist', 'config/devices.cfg')
    DTYPE = np.float32

    def __init__(self, data, meta, parameters, all_parameters=None,
                 global_meta=None):
        self._data = data
        self._meta = meta
        self._parameters = parameters
        if all_parameters is None:
            all_parameters = parameters.copy()
        self._all_parameters = all_parameters
        if global_meta is None:
            global_meta = dict()
        self.global_meta = global_meta

    @property
    def data(self):
        return self._data

    @property
    def meta(self):
        return self._meta

    @property
    def parameters(self):
        return self._parameters

    @property
    def all_parameters(self):
        return self._all_parameters

    def dense(self):
        """
        Return a zero padded dense array of the data
        """
        max_length = self.meta.length.max()
        n_params = len(self.parameters)
        container = np.zeros(
            (len(self.data), max_length, n_params), dtype=self.DTYPE)
        for i, item in enumerate(self.data):
            m, n = item.shape
            container[i, :m, :n] = item
        return container

    def dense_mask(self):
        mask = np.zeros((len(self.data), self.meta.length.max()),
                           dtype=np.bool)
        for i, length in enumerate(self.meta.length):
            mask[i, :length] = True
        return mask


    def map(self, fun):
        return np.array([fun(obj) for obj in self.data])

    def transform(self, fun):
        self._data = self.map(fun)
        self._recompute_lengths()
        return self

    def derive(self, derived_parameter_name, fun):
        self.transform(lambda x: np.hstack((x, fun(x).reshape((-1, 1)))))
        self._parameters.append(derived_parameter_name)
        return self

    def __getitem__(self, selection):
        cls = type(self)
        if not isinstance(selection, slice):
            selection = np.array(selection)
        return cls(
            self.data[selection],
            self.meta[selection].reset_index(drop=True),
            self.parameters,
            self.all_parameters,
            self.global_meta
        )

    def split(self, lprop, column=None):
        """
        Splits into different training sets

        The proportion of rows in the left set is lprop. If column is specified
        selection proportion will be on that column in the metadata.
        """
        if column is None:
            column = self.meta.index
        else:
            column = self.meta[column]

        unique = list(column.unique())
        sample = set(random.sample(unique, int(len(unique) * lprop)))

        selection = column.isin(sample)
        left = self[selection]
        right = self[-selection]
        return (left, right)

    def save(self, outfile):
        meta = np.array([
            self.meta,
            self.parameters,
            self.all_parameters,
            self.global_meta
        ])
        return np.savez_compressed(outfile, data=self.data, meta=meta)

    @classmethod
    def load(cls, infile):
        archive = np.load(infile)
        data = archive['data']
        meta, parameters, all_parameters, global_meta = archive['meta']
        return cls(data, meta, parameters, all_parameters, global_meta)

    @classmethod
    def create(cls, name, device, cyclists=None, parameters=None,
               sessions_config=None, devices_config=None):
        """
        Constructs the dataset and filters out cyclists and parameters

        Parameters
        ----------
        name : str
            Name of dataset to construct
        device : str
            Name of device used to collect data
        cyclists : list of str or None
            List of cyclists to extract. Take all found cyclists if None.
        parameters : list of str or None
            List of parameters to extract. Must correspond to those available
            according to devices config. Take all available if None.
        sessions_config, devices_config : str
            Path to config files for sessions and devices
        """
        # Load config files
        if sessions_config is None:
            sessions_config = cls.SESSIONS_CONFIG
        if devices_config is None:
            devices_config = cls.DEVICES_CONFIG
        sessions = configparser.ConfigParser()
        sessions.read(sessions_config)
        devices = configparser.ConfigParser()
        devices.read(devices_config)

        # Load sessions and device config
        dataset_path = sessions[name]['path']
        meta_matrix = sessions[name]['meta_matrix']
        meta_parameters_path = pkg_resources.resource_filename(
            'procyclist', sessions[name]['meta_parameters'])
        device_matrix = devices[device]['matrix']
        device_filename = devices[device]['filename']

        # Load device parameters from csv file
        parameters_path = pkg_resources.resource_filename(
            'procyclist', devices[device]['parameters'])
        parameters_csv = pd.read_csv(parameters_path, header=None)
        all_parameters = list(parameters_csv[0])

        # Load metadata parameters from csv file
        meta_parameters_csv = pd.read_csv(meta_parameters_path, header=None)
        meta_parameters = list(meta_parameters_csv[0])

        # Select cyclists and parameters
        if cyclists is None:
            cyclists = os.listdir(dataset_path)
        if parameters is None:
            parameters = all_parameters
        parameters_idx = list(map(all_parameters.index, parameters))

        # Extract data and create object
        meta, data = cls._extract(
            dataset_path,
            cyclists,
            device_filename,
            device_matrix,
            parameters_idx,
            meta_matrix,
            meta_parameters
        )
        return cls(data, meta, parameters, all_parameters)

    @classmethod
    def _extract(cls, dataset_path, cyclists, device_filename, device_matrix,
                 parameters_idx, meta_matrix, meta_parameters):
        """
        Extracts all selected parameters from the extracted cyclists

        Assumes that if parameters are missing they will be missing from the end
        """
        meta = list()
        data = list()
        for cyclist in cyclists:
            glob_path = os.path.join(
                dataset_path, cyclist, 'data', device_filename
            )
            for path in glob.glob(glob_path):
                try:
                    mat = sio.loadmat(path)[device_matrix]
                except Exception:
                    print('Data matrix could not be read in: %s' % path)
                    continue
                m, n = mat.shape
                id_ = int(re.findall('\d+', path)[-1])
                try:
                    data.append(mat[:, parameters_idx[:n]].astype(cls.DTYPE))
                except IndexError:
                    print('Data matrix lacking parameters in: %s' % path)
                    continue
                meta.append(dict(cyclist=cyclist, path=path, length=m, id=id_))

        metaframe = None
        for cyclist in cyclists:
            # Load metadata for cyclist
            meta_path = os.path.join(dataset_path, cyclist, cyclist + '.mat')
            try:
                meta_mat = sio.loadmat(meta_path)[cyclist][meta_matrix][0][0]
            except Exception:
                continue
            meta_mat = meta_mat[:, :len(meta_parameters)]
            # Create metadata for cyclist adding cyclist and id columns
            cyclist_meta = pd.DataFrame(
                meta_mat,
                columns=meta_parameters[:meta_mat.shape[1]]
            )
            cyclist_meta['cyclist'] = cyclist
            cyclist_meta['id'] = cyclist_meta.index
            if metaframe is None:
                metaframe = cyclist_meta
            else:
                metaframe = metaframe.append(cyclist_meta)

        # Join metadata for cyclist and set index
        meta = pd.DataFrame(meta)
        meta['order'] = meta.index
        meta = pd.merge(meta, metaframe, on=['cyclist', 'id'])
        meta.sort_values('order', inplace=True)
        meta.set_index('order', drop=True, inplace=True)

        # Select data with corresponding meta data
        data = np.array(data)
        return meta, data[meta.index]

    def _recompute_lengths(self):
        self.meta.length = self.map(lambda mat: mat.shape[0])
