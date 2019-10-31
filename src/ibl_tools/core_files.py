import os
import pandas as pd
import numpy as np
from ibl_tools.core_classes import TrackedGroup
from ibl_tools.util_files import (
    get_dir_files,
    search_str_files,
    extract_group_to_individual_class,
    get_bodyparts_attributes_as_array,
    filter_str_from_str_list,
    GetAttributeList,
)

from ibl_tools.util import get_area_inscribed_quadrilateral_from_coordinates
from moviepy.editor import VideoFileClip
from functools import reduce
from collections import defaultdict

# Path where data is located
BASEPATH = ''
IBL_BASEPATH = os.path.join(BASEPATH, "data/ibl/dlc-networks")
IBL_OUTPATH = os.path.join("/data/model_mpaw")

DIR_PARTS = {
    "paw": "paws-mic-2019-04-26/videos_small/",
    "pupil": "eye-mic-2019-04-16/videos",
    "nostril": "nostril-mic-2019-04-22/videos",
    "tongue": "tongue-mic-2019-04-26/videos",
}


def load_subjects_as_dataframe():
    frames = []
    for part in DIR_PARTS.keys():
        print(part)
        FULLPATH = os.path.join(IBL_BASEPATH, DIR_PARTS[part])
        # get files in directory
        filenames = get_dir_files(FULLPATH)
        files = search_str_files(filenames, ".h5")
        subjects = [
            file.split("_")[2].split(".")[2]
            for file in files
            if file.split("_")[1] == "iblrig"
        ]
        data_nested = pd.DataFrame({"name": subjects, part: True})
        frames.append(data_nested)
        datas = reduce(lambda x, y: pd.merge(x, y, on="name", how="outer"), frames)

    return datas


def load_features_as_dict_from_subject_names(subject_names, subject_ids, body_part):

    train_datas, train_slices = defaultdict(list), defaultdict(list)
    val_datas, val_slices = defaultdict(list), defaultdict(list)
    test_datas, test_slices = defaultdict(list), defaultdict(list)

    for ii, subject_name in zip(subject_ids, subject_names):
        # subject_name = subset_subjects['name'][ii]
        part_fouts = FeatureFiles(subject_name, body_part)

        (train_, val_, test_) = part_fouts.load_trace_features()
        (train_data, train_slice) = train_
        (val_data, val_slice) = val_
        (test_data, test_slice) = test_

        train_datas[ii] = []
        for data in train_data:
            train_datas[ii].append(data)

        val_datas[ii] = []
        for data in val_data:
            val_datas[ii].append(data)

        test_datas[ii] = []
        for data in test_data:
            test_datas[ii].append(data)

    print("Train sets")
    for data_id, datalist in train_datas.items():
        print("{} : {} sets".format(data_id, len(datalist)))

    print("Val sets")
    for data_id, datalist in val_datas.items():
        print("{} : {} sets".format(data_id, len(datalist)))

    print("Test sets")
    for data_id, datalist in test_datas.items():
        print("{} : {} sets".format(data_id, len(datalist)))
    return train_datas, val_datas, test_datas


class TrialFiles(object):
    # Experiment name to setup
    def __init__(self, subject_name, body_part):
        self.DIR_INPUT = os.path.join(IBL_BASEPATH, DIR_PARTS[body_part])
        self.SUBJECT = subject_name

    def get_subject_filenames(self):
        FNAMES_DIR_INPUT = get_dir_files(self.DIR_INPUT)
        subject_filenames = filter_str_from_str_list(
            self.SUBJECT, FNAMES_DIR_INPUT, unique=False
        )

        assert len(subject_filenames) == 4

        self.subject_filenames = subject_filenames
        return self.subject_filenames

    def get_metadata_filename(self, extension=".h5"):
        filenames = self.get_subject_filenames()
        FNAME_METADATA = filter_str_from_str_list(extension, filenames)

        self.FNAME_METADATA = os.path.join(self.DIR_INPUT, FNAME_METADATA)
        return self.FNAME_METADATA

    def get_pickledata_filename(self, extension=".pickle"):
        filenames = self.get_subject_filenames()
        FNAME_PICKLEDATA = filter_str_from_str_list(extension, filenames)
        self.FNAME_PICKLEDATA = os.path.join(self.DIR_INPUT, FNAME_PICKLEDATA)
        return self.FNAME_PICKLEDATA

    def get_video_raw_fname(self, extension=".mp4"):
        filenames = self.get_subject_filenames()
        VIDEO_FILES = filter_str_from_str_list(extension, filenames, unique=False)
        assert len(VIDEO_FILES) <= 2
        for video_file in VIDEO_FILES:
            if not ("label" in video_file):
                FNAME_VIDEO = video_file
            else:
                FNAME_VIDEO_LABELED = video_file

        self.FNAME_VIDEO = os.path.join(self.DIR_INPUT, FNAME_VIDEO)

        if len(VIDEO_FILES) > 1:
            self.FNAME_VIDEO_LABELED = os.path.join(self.DIR_INPUT, FNAME_VIDEO_LABELED)

        return self.FNAME_VIDEO

    # Load raw video
    def load_video_raw(self):
        FILENAME_RAW_VIDEO = self.get_video_raw_fname()

        clip = VideoFileClip(FILENAME_RAW_VIDEO)

        return clip


class FeatureFiles(TrialFiles):
    # Experiment name to setup
    def __init__(self, subject_name, body_part, DIR_OUTPUT=None):
        super().__init__(subject_name, body_part)
        if DIR_OUTPUT == None:
            self.DIR_OUTPUT = IBL_OUTPATH
        else:
            self.DIR_OUTPUT = DIR_OUTPUT
        self.DIR_OUTPUT_SUBJECT = os.path.join(self.DIR_OUTPUT, body_part, subject_name)
        self.DIR_PREPROCESS = os.path.join(self.DIR_OUTPUT_SUBJECT, "preprocessed/")
        self.DIR_FEATURE = os.path.join(self.DIR_OUTPUT_SUBJECT, "features/")

        # Set directory and filenames for output files
        self.FNAME_PREPROCESS = os.path.join(self.DIR_PREPROCESS, "trace.npy")

        self.FNAME_FEATURE = os.path.join(self.DIR_FEATURE, "trace.npy")

        # create folders
        if not os.path.isdir(self.DIR_PREPROCESS):
            os.makedirs(self.DIR_PREPROCESS)

        if not os.path.isdir(self.DIR_FEATURE):
            os.makedirs(self.DIR_FEATURE)

    # Preprocessed data
    def get_trace_preprocessed_fname(self):
        # get clean_data for bodypart
        return self.FNAME_PREPROCESS

    # Features data
    def get_trace_features_fname(self):
        return self.FNAME_FEATURE

    # Preprocessed data
    def load_trace_preprocessed(self, as_mean=False):
        # Load trace preprocessed
        if os.path.isfile(self.FNAME_PREPROCESS):
            data = np.load(self.FNAME_PREPROCESS, allow_pickle=True)
        else:
            return (0, 0, 0)

        if as_mean:
            (xr, yr, name) = data
            mean_x = np.nanmean(xr, 1)
            mean_y = np.nanmean(yr, 1)
            return (mean_x, mean_y, name)
        else:
            return data

    def load_trace_features(self):

        if os.path.isfile(self.FNAME_FEATURE):
            (train_, val_, test_) = np.load(self.FNAME_FEATURE, allow_pickle=True)
        else:
            return (0, 0, 0)

        # (train_data, train_slice) = train_
        # (val_data, val_slice) = val_
        # (test_data, test_slice) = test_
        return (train_, val_, test_)


class TrialFilesPupil(TrialFiles):
    # Experiment name to setup
    def __init__(self, subject_name, body_part):
        super().__init__(subject_name, body_part)
        self.group_string = r"(pupil)"

    def load_trace_raw(self, as_array=True, impose_order=True, verbose=True):
        df = pd.read_hdf(self.get_metadata_filename())
        df_body_parts = df.columns.levels[1]

        # as list
        Pupil = extract_group_to_individual_class(
            df, self.group_string, df_body_parts, verbose=False
        )

        # reorder parts according to fixed order
        if impose_order:
            assert len(Pupil) == 4
            sides = ["top", "bottom", "left", "right"]
            list_strings = GetAttributeList(Pupil, "name")
            indicator_vector = np.zeros(4).astype("int")
            for ss, side in enumerate(sides):
                for part, string in enumerate(list_strings):
                    if side in string:
                        indicator_vector[ss] = part
            # reorder
            tmp = [Pupil[ii] for ii in indicator_vector]
            Pupil = tmp

        if verbose:
            print(GetAttributeList(Pupil, "name"))

        # as array
        pupil = TrackedGroup(*get_bodyparts_attributes_as_array(Pupil))

        if as_array:
            namer, xr, yr, lr = (
                pupil.name,
                pupil.x.copy(),
                pupil.y.copy(),
                pupil.likelihood.copy(),
            )
            return (xr, yr, lr, namer)
        else:
            return Pupil

    def get_pupil_quad_area(self):
        # get coordinates in specified order
        # must impose order
        (x, y, l, name) = self.load_trace_raw(impose_order=True)
        return get_area_inscribed_quadrilateral_from_coordinates(x, y)


class TrialFilesPaw(TrialFiles):

    # Experiment name to setup
    def __init__(self, subject_name, body_part="paw"):
        super().__init__(subject_name, body_part)
        self.group_string = r"(\w+finger_r|pinky_r)"

    def load_trace_raw(self, as_array=True, verbose=True):
        # Load DLC raw traces
        df = pd.read_hdf(self.get_metadata_filename())
        df_body_parts = df.columns.levels[1]

        RightPaw = extract_group_to_individual_class(
            df, self.group_string, df_body_parts, verbose=False
        )

        if verbose:
            print(GetAttributeList(RightPaw, "name"))

        right_paw = TrackedGroup(*get_bodyparts_attributes_as_array(RightPaw))

        if as_array:
            namer, xr, yr, lr = (
                right_paw.name,
                right_paw.x.copy(),
                right_paw.y.copy(),
                right_paw.likelihood.copy(),
            )
            return (xr, yr, lr, namer)
        else:
            return RightPaw
