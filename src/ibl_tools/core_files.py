import os
import re
import pandas as pd
import h5py
import numpy as np
from ibl_tools.core_classes import TrackedGroup
from ibl_tools.util_files import (
    get_dir_files,
    search_str_files,
    extract_group_to_individual_class,
    get_attributes_class_list,
)

from moviepy.editor import VideoFileClip


class ExperimentNames(object):
    """
    Experiment name to setup 
    """

    def __init__(self):
        # directory where experiments are stored
        self.DIR_INPUT = "/data/ibl/dlc-networks"
        self.DIR_PART = "paws-mic-2019-04-26/videos_small"

        # extensions of video files and tracking data
        self.VIDEO_EXTENSION = "_small.mp4"
        self.DLC_EXTENSION = ".h5"

        # -------------------
        # Default assignments
        self.DIR_OUTPUT = "/data/model1/"
        self.DIR_PREPROCESS = "preprocess/"
        self.DIR_FEATURE = "features/"
        # -------------------

    @property
    def get_raw_dir(self):
        DIR_INPUT = self.DIR_INPUT
        DIR_PART = self.DIR_PART

        self.dir_raw = os.path.join(DIR_INPUT, DIR_PART)

        return self.dir_raw

    @property
    def get_raw_files(self):
        # get raw video filenames and raw dlc filenames
        DIR_RAW = self.get_raw_dir
        DLC_EXTENSION = self.DLC_EXTENSION
        VIDEO_EXTENSION = self.VIDEO_EXTENSION

        FNAMES_RAW = get_dir_files(DIR_RAW)

        # return self.dlc_raw_dir_fnames
        # split for directory for video_fnames and dlc_raw_fnames

        dlc_filenames = list(filter(lambda x: re.search(DLC_EXTENSION, x), FNAMES_RAW))
        dlcnames, dlcextensions = zip(*[os.path.splitext(x) for x in dlc_filenames])

        video_filenames = list(
            filter(lambda x: re.search(VIDEO_EXTENSION, x), NAMES_RAW)
        )
        vnames, vextensions = zip(*[os.path.splitext(x) for x in video_filenames])

        # assert we will find traces
        sorted_dlc_names = []
        for video_name in vnames:
            dlc_name_match = list(filter(lambda x: re.search(video_name, x), dlcnames))
            assert len(dlc_name_match) == 1
            sorted_dlc_names.append(dlc_name_match[0])

        # self.FILENAME_DLC_RAW = sorted_dlc_names
        # self.FILENAME_VIDEO = vnames
        file_names = []
        for video_name in vnames:
            file_names.append(video_name.split(".")[2])
        return file_names


class ExperimentFiles(object):
    def __init__(self, experiment_name, subject_id):
        self.EXPERIMENT_NAME = experiment_name
        # directory where experiments are stored
        self.DIR_INPUT = "/data/ibl/dlc-networks"

        # invididual body part
        self.DIR_PART = "paws-mic-2019-04-26/videos_small"
        # side (part) of body
        self.DIR_BODY_PART = "right_paw/"

        # default file extensions from input data
        self.VIDEO_EXTENSION = "_small.mp4"
        self.DLC_FNAME_EXTENSION = ".h5"
        self.DLC_FNAME_PRE_EXTENSION = "_small.h5"

        # -------------------
        # Defaul assignments
        # -------------------

        # base directory of data
        self.DIR_OUTPUT = "/data/model_01/subject_{:02}/".format(subject_id)
        # directory to store preprocessed data
        self.DIR_PREPROCESS = "preprocess/"
        # directory to store features to input model
        self.DIR_FEATURE = "features/"

        # set filename for clean body part
        self.fname_preprocessed = "paw_traces.npy"

        # set filename for features
        self.fname_features = "features.npy"
        # Set directory for preprocessed traces
        self.DIR_OUTPUT_PREPROCESS = os.path.join(
            self.DIR_OUTPUT, self.DIR_PREPROCESS, self.DIR_BODY_PART
        )

        # Set directory for features from traces
        self.DIR_OUTPUT_FEATURE = os.path.join(
            self.DIR_OUTPUT, self.DIR_FEATURE, self.DIR_BODY_PART
        )

        # make data for video
        if not os.path.isdir(self.DIR_OUTPUT_PREPROCESS):
            os.makedirs(self.DIR_OUTPUT_PREPROCESS)

        if not os.path.isdir(self.DIR_OUTPUT_FEATURE):
            os.makedirs(self.DIR_OUTPUT_FEATURE)

        # Set directory and filenames for input files
        self.set_raw_fnames()

        # Set directory and filenames for output files
        self.set_out_fnames()

    def set_out_fnames(self):
        fname = os.path.join(self.DIR_OUTPUT_PREPROCESS, self.fname_preprocessed)
        self.FILE_PREPROCESS = fname

        fname = os.path.join(self.DIR_OUTPUT_FEATURE, self.fname_features)
        self.FILE_FEATURES = fname

    def set_raw_fnames(self):
        DIR_INPUT_DLC = os.path.join(self.DIR_INPUT, self.DIR_PART)
        DIR_INPUT_DLC_FILES = get_dir_files(DIR_INPUT_DLC)

        # Files in directory associated with experiment
        raw_filenames = search_str_files(
            DIR_INPUT_DLC_FILES, str_match=self.EXPERIMENT_NAME
        )

        # Raw video filename
        video_fname = search_str_files(raw_filenames, str_match=self.VIDEO_EXTENSION)
        assert len(video_fname) == 1

        self.VIDEO_FNAME = os.path.join(DIR_INPUT_DLC, video_fname[0])

        # Raw DLC trace filename
        dlc_fname = search_str_files(raw_filenames, str_match=self.DLC_FNAME_EXTENSION)
        assert len(dlc_fname) == 1

        self.DLC_FNAME = os.path.join(DIR_INPUT_DLC, dlc_fname[0])

    def get_video_raw_fname(self):
        # get the fullpath where video is located
        assert os.path.isfile(self.VIDEO_FNAME)
        return self.VIDEO_FNAME

    def get_trace_raw_fname(self):
        # get the fullpath where traces are located
        assert os.path.isfile(self.DLC_FNAME)
        return self.DLC_FNAME

    # Preprocessed data
    def get_trace_preprocessed_fname(self):
        # get clean_data for bodypart
        return self.FILE_PREPROCESS

    # Features data
    def get_trace_features_fname(self):
        return self.FILE_FEATURES

    # Load raw video
    def load_video_raw(self):
        FILENAME_RAW_VIDEO = self.get_video_raw_fname()

        clip = VideoFileClip(FILENAME_RAW_VIDEO)

        return clip

    def load_trace_raw(self, as_array=True):
        # Load DLC raw traces
        DLC_RAW_FNAME = self.get_trace_raw_fname()

        # hd5
        df = pd.read_hdf(DLC_RAW_FNAME)
        df_body_parts = df.columns.levels[1]

        if "right" in self.DIR_BODY_PART:
            RightPaw = extract_group_to_individual_class(
                df, r"(\w+finger_r|pinky_r)", df_body_parts
            )
        else:
            raise ("We don" "t have that data")

        right_paw = TrackedGroup(*get_attributes_class_list(RightPaw))

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

    def load_trace_preprocessed(self, as_mean=True):
        # Load trace preprocessed
        if os.path.isfile(self.FILE_PREPROCESS):
            data = np.load(self.FILE_PREPROCESS, allow_pickle=True)
        else:
            return (0, 0, 0)

        # (xr, yr, name) = data
        # xr T x D
        if as_mean:
            (xr, yr, name) = data
            mean_x = np.nanmean(xr, 1)
            mean_y = np.nanmean(yr, 1)
            return (mean_x, mean_y, name)
        else:
            return data

    def load_trace_features(self, as_mean=True):

        if os.path.isfile(self.FILE_FEATURES):
            (train_, val_, test_) = np.load(self.FILE_FEATURES, allow_pickle=True)
        else:
            return (0, 0, 0)

        # (train_data, train_slice) = train_
        # (val_data, val_slice) = val_
        # (test_data, test_slice) = test_
        return (train_, val_, test_)
