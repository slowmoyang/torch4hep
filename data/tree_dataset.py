from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from torch.utils.data import Dataset

import ROOT


class TreeDataset(Dataset):
    def __init__(self, path, tree_name, loading_baskets=False):
        self._root_file = ROOT.TFile.Open(path, "READ")
        self._tree = self._root_file.Get(tree_name)
        self._path = path
        self._tree_name = tree_name

        # FIXME too little performcne improvements..
        # Performance improvements are not sufficient when using shuffle.
        if loading_baskets:
            self.load_baskets()

    def __len__(self):
        return int(self._tree.GetEntries())

    def __getitem__(self, idx):
        raise NotImplementedError

    def load_baskets(self, branches=None):
        if branches is None:
            self._tree.LoadBaskets()
        else:
            raise NotImplementedError
