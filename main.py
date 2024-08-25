import cv2
import pickle
import numpy as np
import step01_annotate_image.utils_general as utils_general
from step01_annotate_image.annotate_image import FrameAnnotatorPose, FrameAnnotatorPoseUtils
from step03_parse_image.parse_image import crop_pedestrians

# Initialize Detection Targets
targets = utils_general.get_detection_targets()

# Initialize Frame Annotator Tools
fa_pose_utils = FrameAnnotatorPoseUtils()
fa_pose = FrameAnnotatorPose(
    general_utils=utils_general,
    annotator_utils=fa_pose_utils
)

# Get Mediapipe Model Instance
mp_drawing, mp_pose = fa_pose_utils.init_mp()

""" Models """
# Initialize Mediapipe Model
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Self-trained Classification Model
with open("./data/models/posture_classify.pkl", "rb") as f:
    stc_model = pickle.load(f)

with open("./data/models/posture_classify_scaler.pkl", "rb") as fs:
    stc_model_scaler = pickle.load(fs)


# Process One Frame
def process_one_frame(frame):
    # Crop out pedestrians
    pedestrian_frames = crop_pedestrians(frame)

    predictions = []

    # Process each subframe
    for pedestrian_frame in pedestrian_frames:
        pedestrian_frame = cv2.cvtColor(pedestrian_frame, cv2.COLOR_RGB2BGR)
        key_coord_angles, pose_results = fa_pose.process_one_frame(
            pedestrian_frame,
            targets=targets,
            model=pose
        )
        if key_coord_angles is None:
            predictions.append("unknown")
            continue

        _numeric_data = np.array([kka['angle'] for kka in key_coord_angles]).reshape(1, -1)
        numeric_data = stc_model_scaler.transform(_numeric_data)

        prediction = stc_model.predict(numeric_data)
        match prediction:
            case 0:
                text = "not using"
            case 1:
                text = "using"
            case _:
                text = "unknown"

        predictions.append(text)
    return predictions


if __name__ == "__main__":
    frame = cv2.imread("./data/test_parse_image/_test/test_img.png")
    predictions = process_one_frame(frame)
    for prediction in predictions:
        print(prediction)
