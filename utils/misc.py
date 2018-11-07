from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import warnings


class Directory(object):
    def __init__(self, path, creation=True):
        self.path = path
        self._creation = creation
        if self._creation:
            os.makedirs(self.path)

    def make_subdir(self, name):
        path = os.path.join(self.path, name)
        setattr(self, name, Directory(path, creation=self._creation))

    def get_entries(self, full_path=True):
        entries = os.listdir(self.path)
        if full_path:
            entries = [os.path.join(self.path, each) for each in entries]
        return entries


def is_float(string):
    if not isinstance(string, str):
        raise TypeError
    return re.match("^\d+?\.\d+?$", string) is not None


def convert_str_to_number(string, warning=True):
    if not isinstance(string, str):
        raise TypeError

    if is_float(string):
        return float(string)
    elif string.isdigit():
        return int(string)
    else:
        if warning:
            warnings.warn("'{}' is neither 'float' nor 'int'".format(string),
                          UserWarning)
        return string
