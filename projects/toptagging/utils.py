from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob

import pandas as pd

from torch4hep.utils.misc import convert_str_to_number


def find_good_state(directory,
                    which={"max": ["auc"], "min": ["loss"]},
                    verbose=True,
                    extension=".pth.tar"):
    """
    path: '/path/to/directory/<MODEL NAME>_loss-0.2345_<KEY>-<VALUE>.pth.tar'
    """
    def _parse_path(path): 
        basename = os.path.basename(path)
        basename = basename.replace(extension, "")

        metadata = basename.split("_")[1:]
        metadata = [each.split("-") for each in metadata]
        metadata = {key: convert_str_to_number(value, warning=False) for (key, value) in metadata}
        metadata["path"] = path
        return metadata

    entries = glob.glob(os.path.join(directory, "*.pth.tar"))
    entries = [_parse_path(each) for each in entries if each.endswith(extension)]
    df = pd.DataFrame(entries)

    good_models = []
    for each in which["max"]:
        path = df.loc[df[each].idxmax()]["path"]
        good_models.append(path)
        if verbose:
            print("Max {}: {}".format(each, path))
    for each in which["min"]:
        good_models.append(df.loc[df[each].idxmin()]["path"])
        if verbose:
            print("Min {}: {}".format(each, path))
    good_models = list(set(good_models))
    return good_models        
