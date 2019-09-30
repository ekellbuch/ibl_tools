"""
Files to load ibl data
"""

import re
import os
import numpy as np
from ibl_tools.core_classes import TrackedPoint


def extract_group_to_individual_class(df, parts_group, df_body_parts):

    parts_group_label = list(filter(lambda x: re.match(parts_group, x), df_body_parts))

    print(parts_group_label)
    print(
        "\nParts in group: ' {} ': \n".format(parts_group)
        + "\n".join(parts_group_label)
    )

    g = lambda bodypart: TrackedPoint(bodypart, *read_vals_key(df, bodypart))

    return list(map(g, parts_group_label))


def read_vals_key(df, body_part):
    # s=df[(df.keys()[0][0], 'pupil_top_r', 'x')].values
    x = df[df.keys()[0][0], body_part, "x"].values
    y = df[df.keys()[0][0], body_part, "y"].values
    likelihood = df[df.keys()[0][0], body_part, "likelihood"].values
    return x, y, likelihood



def get_dir_files(dir):
    """
    Get files in directory
    Get files in directory
    Args:
        dir:
    Returns:
    """
    onlyfiles = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    return onlyfiles


def search_str_files(files, str_match="", split_extension=False):
    filenames = list(filter(lambda x: re.search(str_match, x), files))

    if split_extension:
        fnames, fextensions = zip(*[os.path.splitext(x) for x in filenames])
        return fnames, fextensions
    else:
        return filenames



def GetAttributeList(Group, attribute):
    return np.asarray([getattr(part, attribute) for part in Group])


def get_attributes_class_list(parts):
    xr = GetAttributeList(parts, "x")
    yr = GetAttributeList(parts, "y")
    llr = GetAttributeList(parts, "likelihood")
    names_r = GetAttributeList(parts, "name")
    return names_r, xr, yr, llr
