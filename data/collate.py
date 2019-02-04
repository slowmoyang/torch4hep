from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re

import torch
from torch._six import container_abcs
from torch._six import string_classes
from torch._six import int_classes
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

try:
    from torch.utils.data.dataloader import numpy_type_map as NUMPY_TYPE_MAP
except ImportError:
    import warnings
    from torch.utils.data._utils.collate import numpy_type_map as NUMPY_TYPE_MAP
    # TODO warnings


class VarLenCollator(object):
    def __init__(self,
                 padding=[],
                 packing=[],
                 batch_first=True):
        '''
        Arguments
        '''
        padding = set(padding)
        packing = set(packing)

        assert len(padding.intersection(packing)) == 0

        self.padding = padding
        self.packing = packing
        self.batch_first = batch_first

    def __call__(self, batch, key=None):
        '''
        Arguments:
            batch (sequence)
        '''
        elem_type = type(batch[0])
        if isinstance(batch[0], torch.Tensor):
            if key in self.padding:
                return self.pad(batch)
            elif key in self.packing:
                return self.pack(batch)
            else:
                return torch.stack(batch, 0)
        elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
                and elem_type.__name__ != 'string_':
            elem = batch[0]
            if elem_type.__name__ == 'ndarray':
                # array of string classes and object
                if re.compile(r'[SaUO]').search(elem.dtype.str) is not None:
                    raise TypeError(error_msg_fmt.format(elem.dtype))

                return self.__call__([torch.from_numpy(b) for b in batch], key=key)
            if elem.shape == ():  # scalars
                py_type = float if elem.dtype.name.startswith('float') else int
                return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
        elif isinstance(batch[0], float):
            return torch.tensor(batch, dtype=torch.float64)
        elif isinstance(batch[0], int_classes):
            return torch.tensor(batch)
        elif isinstance(batch[0], string_classes):
            return batch
        elif isinstance(batch[0], container_abcs.Mapping):
            return {key: self.__call__([each[key] for each in batch], key=key) for key in batch[0]}
        elif isinstance(batch[0], container_abcs.Sequence):
            transposed = zip(*batch)
            return [self.__call__(samples, key=key) for samples in transposed]

    def pad(self, batch):
        return pad_sequence(batch, batch_first=self.batch_first)

    def pack(self, batch):
        lengths = torch.LongTensor([len(each) for each in batch])
        batch = pad_sequence(batch, batch_first=self.batch_first)
        lengths, indices = lengths.sort(0, descending=True)
        batch = batch[indices]
        return pack_padded_sequence(batch, lengths, batch_first=self.batch_first)


