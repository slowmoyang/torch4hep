from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

import pickle
import numpy as np
from datetime import datetime


from torch4hep.datasets import BaseTreeDataset


class Normalizer(object):
    def __init__(self, dset, ban=[], verbose=True):
        if isinstance(dset, BaseTreeDataset):
            print("[{}] Start to analyse dataset".format(datetime.now()))
            self._keys = [key for key in dset[0].keys() if not key in ban]
            bucket = {key: [] for key in self._keys}
            num_examples = len(dset)
            for i in xrange(num_examples):
                example = dset[i]
                for key in self._keys:
                    bucket[key].append(example[key])
            bucket = {key: np.array(value) for key, value in bucket.iteritems()} 
            mean = {key: value.mean(axis=0) for key, value in bucket.iteritems()}
            # centering
            bucket = {key: value - mean[key] for key, value in bucket.iteritems()}
            std = {key: value.std(axis=0) for key, value in bucket.iteritems()}
            for key, value in std.iteritems():
                is_zero = value == 0
                value[is_zero] = 1e-8
                std[key] = value

            self._mean = mean
            self._std = std
            print("[{}] Finish off analysing dataset".format(datetime.now()))
        elif isinstance(dset, str):
            self.load(dset)
        else:
            # FIXME
            raise ValueError("dset should be BaseTreeDataset or a path to pickle file")
        
    def __call__(self, example):
        for key in self._keys:
            value = example.pop(key)
            value = value - self._mean[key]
            value = value / self._std[key]
            example[key] = value
        return example

    def __repr__(self):
        return self.__class__.__name__ + " ({})".format(
            " ".join(self._keys))

    def save(self, path):
        data = {
            "mean": self._mean,
            "std": self._std
        }

        with open(path, "w") as pickle_file:
            pickle.dump(data, pickle_file)


    def load(self, path):
        with open(path, "r") as pickle_file:
            data = pickle.load(pickle_file)
        self._mean = data["mean"]
        self._std = data["std"]
        self._keys = data["mean"].keys()



