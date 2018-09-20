from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

# TODO axis of seq_len
class SeqLenAdjuster(object):
    def __init__(self,
                 max_len,
                 padding="post",
                 truncating="post",
                 value=0.0): 

        padding = padding.lower()
        truncating = truncating.lower()

        if padding == "pre":
            self._pad = self.pad_pre 
        elif padding == "post":
            self._pad = self.pad_post
        else:
            raise ValueError("'padding' should be 'pre' or 'post'. (case in-sesnsitive)")

        if truncating == "pre":
            self._truncate = self.truncate_pre
        elif truncating == "post":
            self._truncate = self.truncate_post
        else:
            raise ValueError("'truncating' should be 'pre' or 'post'. (case in-sesnsitive)")

        self.max_len = max_len
        self.padding = padding
        self.truncating = truncating
        self.value = value

    def pad_pre(self, sequence):
        seq_len = sequence.shape[0]
        diff = abs(self.max_len - seq_len)
        pad_width = ((diff, 0), (0, 0))
        return np.pad(sequence, pad_width, mode="constant", constant_values=self.value)

    def pad_post(self, sequence):
        seq_len = sequence.shape[0]
        diff = abs(self.max_len - seq_len)
        pad_width = ((0, diff), (0, 0))
        return np.pad(sequence, pad_width, mode="constant", constant_values=self.value)

    def truncate_pre(self, sequence):
        return sequence[-self.max_len: ]

    def truncate_post(self, sequence):
        return sequence[: self.max_len]

    def transform(self, sequence):
        seq_len = sequence.shape[0]
        if seq_len == self.max_len:
            pass
        elif seq_len < self.max_len:
            sequence = self._pad(sequence)
        else:
            sequence = self._truncate(sequence)
        return sequence

    def __call__(self, sequence):
        rank = len(sequence.shape)
        if rank == 1:
            return self.transform(sequence.reshape(-1, 1)).reshape(-1)
        elif rank == 2:
            return self.transform(sequence) 
        else:
            raise ValueError
