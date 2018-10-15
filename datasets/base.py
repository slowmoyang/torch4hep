from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset

import ROOT


class BaseTreeDataset(Dataset):
    def __init__(self, path, tree_name, transform=None):
        self._root_file = ROOT.TFile.Open(path, "READ")
        self._tree = self._root_file.Get(tree_name)
        self._path = path
        self._tree_name = tree_name
        self._transform = transform

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, transform_):
        self._transform = transform_

    def __len__(self):
        return int(self._tree.GetEntries())

    def __getitem__(self, idx):
        raise NotImplementedError
