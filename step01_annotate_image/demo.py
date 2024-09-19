# Packages
import pickle

# Local
from annotate_image import FrameAnnotatorPose
from utils_annotation import FrameAnnotatorPoseUtils
import utils_general as utils

# Initialize Detection Targets
pose_targets = utils.get_detection_targets()

# Initialize Utilities
fa_pose_utils = FrameAnnotatorPoseUtils()

# Inject Utilities to Annotator
fa_pose = FrameAnnotatorPose(general_utils=utils, annotator_utils=fa_pose_utils)

with open("../data/models/posture_dt.pkl", "rb") as f:
    model = pickle.load(f)

with open("../data/models/posture_dt_scaler.pkl", "rb") as fs:
    model_scaler = pickle.load(fs)

cap = utils.init_video_capture(0)
fa_pose.demo(cap, pose_targets, [model, model_scaler])
