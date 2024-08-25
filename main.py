import cv2
import pickle
import torch
import numpy as np
import step01_annotate_image.utils_general as utils_general
from step01_annotate_image.annotate_image import FrameAnnotatorPose, FrameAnnotatorPoseUtils
from step03_parse_image.parse_image import crop_pedestrians


# Process One Frame
def process_one_frame(frame_to_process, pose_model, YOLO_model=None, wait=False):
    # Crop out pedestrians
    pedestrian_frames, xyxy_sets = crop_pedestrians(frame_to_process, model=YOLO_model)

    # Number of people
    num_people = len(pedestrian_frames)

    # Process each subframe
    for pedestrian_frame, xyxy in zip(pedestrian_frames, xyxy_sets):
        pedestrian_frame = cv2.cvtColor(pedestrian_frame, cv2.COLOR_RGB2BGR)
        key_coord_angles, pose_results = fa_pose.process_one_frame(
            pedestrian_frame,
            targets=targets,
            model=pose_model
        )

        if key_coord_angles is None or pedestrian_frames is None:
            # predictions.append("unknown")
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

        # predictions.append(text)

        cv2.putText(
            frame_to_process,
            text,
            (int(xyxy[0]), int(xyxy[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2)
        cv2.rectangle(
            frame_to_process,
            (int(xyxy[0]), int(xyxy[1])),
            (int(xyxy[2]), int(xyxy[3])),
            (0, 255, 0),
            2
        )

        cv2.imshow("Test", frame_to_process)

        if wait:
            cv2.waitKey(90000)
    return num_people


if __name__ == "__main__":

    """ Utilities """
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

    # YOLO Model
    YOLOv5s_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to('cuda')

    """ Video """
    cap = cv2.VideoCapture("./data/test_parse_image/_test/test_video_short.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        num_people = process_one_frame(frame, pose, YOLOv5s_model, wait=False)
        cv2.waitKey(1)
