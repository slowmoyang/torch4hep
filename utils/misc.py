from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import re
import warnings
from datetime import datetime
import logging
import copy


class Directory(object):
    def __init__(self, path, create=True):
        self._path = path
        self._create = create
        if self._create:
            os.makedirs(self.path)

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, path_):
        if not isinstance(path_, str):
            raise TypeError
        self._path = path_

    def mkdir(self, name):
        path = os.path.join(self._path, name)
        setattr(self, name, Directory(path, create=self._create))

    def get_entries(self, full_path=True):
        entries = os.listdir(self.path)
        if full_path:
            entries = [os.path.join(self._path, each) for each in entries]
        return entries

    def concat(self, name):
        return os.path.join(self._path, name)


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


def get_logger(name, path):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    format_str = '[%(asctime)s] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_str, date_format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def parse_str(string, target=None):
    data = string.split("_")
    data = [each.split("-") for each in data if "-" in each]
    data = {key: convert_str_to_number(value) for key, value in data}
    if target is None:
        return data
    else:
        return data[target]
