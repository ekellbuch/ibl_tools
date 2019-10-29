"""
Files to load ibl data
"""

import re
import os
import numpy as np
from ibl_tools.core_classes import TrackedPoint


def extract_group_to_individual_class(df, parts_group, df_body_parts, verbose=True):
    """
    Extract multiple keys from dataframe and combine them in a trackedgroup
    Inputs:
    _______
    df : dataframe
    parts_group : string
    df_body_parts: list of strings
    """
    parts_group_label = list(filter(lambda x: re.match(parts_group, x), df_body_parts))

    if verbose:
        print(parts_group_label)
        print(
        "\nParts in group: ' {} ': \n".format(parts_group)
            + "\n".join(parts_group_label)
        )
    g = lambda bodypart: TrackedPoint(bodypart, *read_vals_key(df, bodypart))

    return list(map(g, parts_group_label))


def read_vals_key(df, body_part):
    """
    Given dlc dictionary, and part_name (key)
    Extract part name as TrackedPoint class

    Inputs:
    ______
    :param df: pandas dataframe
        dataframe where body parts are stored
    :param body_part: string
        name of body part
    :return:
        x: array
            x 
        y: array
        likelihood: array
    """
    # s=df[(df.keys()[0][0], 'pupil_top_r', 'x')].values
    x = df[df.keys()[0][0], body_part, "x"].values
    y = df[df.keys()[0][0], body_part, "y"].values
    likelihood = df[df.keys()[0][0], body_part, "likelihood"].values
    return x, y, likelihood


def get_dir_files(dir):
    """
    Get files in directory
    Args:
        dir: directory
    Returns: 
    
    filenames:
        files in directory
    """
    filenames = [f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))]

    return filenames


def search_str_files(files, str_match="", split_extension=False):
    filenames = list(filter(lambda x: re.search(str_match, x), files))

    if split_extension:
        fnames, fextensions = zip(*[os.path.splitext(x) for x in filenames])
        return fnames, fextensions
    else:
        return filenames


def GetAttributeList(Group, attribute):
    """
    Extract attribute for elements in Group
    :param Group:
    :param attribute:
    :return:
    """
    return np.asarray([getattr(part, attribute) for part in Group])


def get_bodyparts_attributes_as_array(parts):
    """
    Extract x, y, likelihood and name keys from parts
    as arrays
    :param parts:
    :return:
    """
    xr = GetAttributeList(parts, "x")
    yr = GetAttributeList(parts, "y")
    llr = GetAttributeList(parts, "likelihood")
    names_r = GetAttributeList(parts, "name")
    return names_r, xr, yr, llr


#%%
def assign_bodypart_to_class(dlc_dict, part_name):
    """
    Given dlc dictionary, and part_name (key)
    Extract part name as TrackedPoint class
    :param dlc_dict:
    :param part_name:
    :return:
    """
    x = dlc_dict[part_name + "_x"]
    y = dlc_dict[part_name + "_y"]
    likelihood_dlc = dlc_dict[part_name + "_likelihood"]

    assert x.size == y.size == likelihood_dlc.size
    return TrackedPoint(part_name, x, y, likelihood_dlc)


def get_group_bodyparts(
    dlc_dict,
    parts_group=["pinky_r", "middle_finger_r", "ring_finger_r", "pointer_finger_r"],
):
    """
    Get body parts as list for a group given dlc_dict
    :param dlc_dict:
    :param parts_group:
    :return:
    """
    return [assign_bodypart_to_class(dlc_dict, part) for part in parts_group]


def load_dlc(folder_path, camera="left"):
    import alf.io
    """
    Load in DLC traces and timestamps from FPGA and align them

    Parameters
    ----------
    folder_path: string of the path to the top-level folder of recording
    camera: which camera to use ('left', 'right', 'bottom')
    
    -----------------------
    Function extraced from
    https://github.com/int-brain-lab/ibllib
    """

    # Load in DLC data
    dlc_dict = alf.io.load_object(
        os.path.join(folder_path, "alf"), "_ibl_%sCamera" % camera
    )
    dlc_dict["camera"] = camera
    dlc_dict["units"] = "px"

    # Hard-coded hack because extraction of timestamps was wrong
    if camera == "left":
        camera = "body"

    # Load in FPGA timestamps
    timestamps = np.load(
        os.path.join(
            folder_path, "raw_video_data", "_iblrig_%sCamera.times.npy" % camera
        )
    )

    # Align FPGA and DLC timestamps
    if len(timestamps) > len(dlc_dict[list(dlc_dict.keys())[0]]):
        timestamps = timestamps[0 : len(dlc_dict[list(dlc_dict.keys())[0]])]
    elif len(timestamps) < len(dlc_dict[list(dlc_dict.keys())[0]]):
        for key in list(dlc_dict.keys()):
            dlc_dict[key] = dlc_dict[key][0 : len(timestamps)]
    dlc_dict["timestamps"] = timestamps
    dlc_dict["sampling_rate"] = 1 / np.mean(np.diff(timestamps))

    return dlc_dict



def filter_str_from_str_list(string_filter, list_strings, unique=True):
    """
    Filter strings in list based on my_string
    similar to list(filter(lambda x: re.search(subject_name, x), FNAMES_RAW))
    
    """
    string_matches = list(filter(lambda k: string_filter in k, list_strings))
    
    if unique:
        assert len(string_matches) == 1
        string_matches = string_matches[0]

    return string_matches

