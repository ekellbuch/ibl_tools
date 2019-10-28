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
    get_bodyparts_attributes_as_array,
    filter_str_from_str_list,
    GetAttributeList,
)

from ibl_tools.util import get_area_inscribed_quadrilateral_from_coordinates
from moviepy.editor import VideoFileClip


# Path where data is located
BASEPATH = ''
IBL_BASEPATH = os.path.join(BASEPATH, 'data/ibl/dlc-networks')

DIR_PARTS = {
    "paw": 'paws-mic-2019-04-26/videos_small/',
    "pupil": 'eye-mic-2019-04-16/videos',
    "nostril": 'nostril-mic-2019-04-22/videos',
    "tongue": 'tongue-mic-2019-04-26/videos',
    
}


class ExperimentNames(object):
    """
    This class only works for the paw position
    TO DO: Match string instead of searching

    """
    #Experiment name to setup 
    def __init__(self):
        # directory where experiments are stored
        self.DIR_INPUT = os.path.join(BASEPATH, "data/ibl/dlc-networks")
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
        self.DIR_INPUT = os.path.join(BASEPATH, "data/ibl/dlc-networks")
        print(self.DIR_INPUT)

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


    
class TrialFiles(object):
    #Experiment name to setup 
    def __init__(self, subject_name, body_part):
        self.DIR_INPUT = os.path.join(IBL_BASEPATH, DIR_PARTS[body_part])
        self.SUBJECT = subject_name

    def get_subject_filenames(self):
        FNAMES_DIR_INPUT = get_dir_files(self.DIR_INPUT)
        subject_filenames = filter_str_from_str_list(self.SUBJECT,
                                                     FNAMES_DIR_INPUT, unique=False)
        
        assert len(subject_filenames) == 4
        
        self.subject_filenames = subject_filenames
        return self.subject_filenames
    
    def get_metadata_filename(self, extension='.h5'):
        filenames = self.get_subject_filenames()
        FNAME_METADATA = filter_str_from_str_list(extension, filenames)
        
        self.FNAME_METADATA = os.path.join(self.DIR_INPUT, FNAME_METADATA)
        return self.FNAME_METADATA

    def get_pickledata_filename(self, extension='.pickle'):
        filenames = self.get_subject_filenames()
        FNAME_PICKLEDATA = filter_str_from_str_list(extension, filenames)
        self.FNAME_PICKLEDATA = os.path.join(self.DIR_INPUT, FNAME_PICKLEDATA)
        return self.FNAME_PICKLEDATA
    
    def get_video_raw_fname(self, extension='.mp4'):
        filenames = self.get_subject_filenames()
        VIDEO_FILES = filter_str_from_str_list(extension,
                                               filenames,
                                               unique=False)
        assert len(VIDEO_FILES) <=2    
        for video_file in VIDEO_FILES:
            if not('label' in  video_file):
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
    #Experiment name to setup 
    def __init__(self, subject_name, body_part, DIR_OUTPUT=None):
        super().__init__(subject_name, body_part)
        if DIR_OUTPUT == None:
            self.DIR_OUTPUT = "/data/model_mpaw/"
        else:
            self.DIR_OUTPUT = DIR_OUTPUT
        self.DIR_OUTPUT_SUBJECT = os.path.join(self.DIR_OUTPUT, body_part, subject_name)
        self.DIR_PREPROCESS = os.path.join(self.DIR_OUTPUT_SUBJECT, "preprocessed/")
        self.DIR_FEATURE = os.path.join(self.DIR_OUTPUT_SUBJECT, "features/")
        
        # Set directory and filenames for output files
        self.FNAME_PREPROCESS = os.path.join(self.DIR_PREPROCESS, 'trace.npy')
        
        self.FNAME_FEATURE = os.path.join(self.DIR_FEATURE, 'trace.npy')

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
            (train_, val_, test_) = np.load(self.FNAME_FEATURE,
                                            allow_pickle=True)
        else:
            return (0, 0, 0)

        # (train_data, train_slice) = train_
        # (val_data, val_slice) = val_
        # (test_data, test_slice) = test_
        return (train_, val_, test_)


class TrialFilesPupil(TrialFiles):

    #Experiment name to setup 
    def __init__(self, subject_name, body_part):
        super().__init__(subject_name, body_part)
        self.group_string=r'(pupil)'
    
    def load_trace_raw(self, as_array=True, impose_order=True, verbose=True):
        df = pd.read_hdf( self.get_metadata_filename())
        df_body_parts = df.columns.levels[1]

        # as list
        Pupil = extract_group_to_individual_class( df, self.group_string, df_body_parts, verbose=False)

        # reorder parts according to fixed order
        if impose_order:
            assert len(Pupil) == 4
            sides = ['top','bottom', 'left', 'right']
            list_strings = GetAttributeList(Pupil, 'name')        
            indicator_vector = np.zeros(4).astype('int')
            for ss, side in enumerate(sides):
                for part, string in enumerate(list_strings):
                    if side in string:
                        indicator_vector[ss] = part
            # reorder
            tmp = [Pupil[ii] for ii in indicator_vector]
            Pupil = tmp
        
        if verbose:
            print(GetAttributeList(Pupil, 'name'))

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

    #Experiment name to setup 
    def __init__(self, subject_name, body_part='paw'):
        super().__init__(subject_name, body_part)
        self.group_string=r"(\w+finger_r|pinky_r)"
    
    def load_trace_raw(self, as_array=True, verbose=True):
        # Load DLC raw traces
        df = pd.read_hdf( self.get_metadata_filename())
        df_body_parts = df.columns.levels[1]

        RightPaw = extract_group_to_individual_class(
            df, self.group_string, df_body_parts, verbose=False
        )
        
        if verbose:
            print(GetAttributeList(RightPaw, 'name'))

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

