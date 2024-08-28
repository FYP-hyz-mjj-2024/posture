# Package
import cv2
import pickle
import torch
import numpy as np
from ultralytics import YOLO

# Local
import step01_annotate_image.utils_general as utils_general
from step01_annotate_image.annotate_image import FrameAnnotatorPose, FrameAnnotatorPoseUtils
from step03_parse_image.parse_image import crop_pedestrians


def process_one_frame(
        frame_to_process,
        stc_model_and_scaler,
        mp_pose_model,
        YOLO_model=None,
        device='cpu'):
    """
    Take a single frame. Use YOLO model to locate all the pedestrians. Then, for each retrieved
    pedestrian, feed it into mediapipe posture detection model to extract landmarks. After that,
    calculate key features from the landmarks, and feed to the self-trained classification model.
    The self-trained classification model will return the prediction results, which will be then
    rendered on the frame.
    :param frame_to_process: The video frame retrieved from cv2.VideoCapture().read()
    :param stc_model_and_scaler: A tuple/array of the self-trained classification model and scaler.
    :param mp_pose_model: The mediapipe posture detection model.
    :param YOLO_model: The YOLO model for pedestrian location & image cropping.
    :param device: GPU support.
    """
    stc_model, stc_model_scaler = stc_model_and_scaler
    # Crop out pedestrians
    pedestrian_frames, xyxy_sets = crop_pedestrians(frame_to_process, model=YOLO_model, device=device)

    # Number of people
    num_people = len(pedestrian_frames)

    if num_people <= 0:
        cv2.imshow("Smartphone Usage Detection", frame_to_process)
        return 0

    # Process each subframe
    for pedestrian_frame, xyxy in zip(pedestrian_frames, xyxy_sets):
        # Get the key angle array from a subframe.
        pedestrian_frame = cv2.cvtColor(pedestrian_frame, cv2.COLOR_RGB2BGR)
        key_coord_angles, pose_results = fa_pose.process_one_frame(
            pedestrian_frame,
            targets=targets,
            model=mp_pose_model
        )

        if key_coord_angles is None or pedestrian_frames is None:
            continue

        _numeric_data = np.array([kka['angle'] for kka in key_coord_angles]).reshape(1, -1)

        # Feed the normalized angle array into the self-trained model, get prediction.
        numeric_data = stc_model_scaler.transform(_numeric_data)
        prediction_boolean = stc_model.predict(numeric_data)
        match prediction_boolean:
            case 0:
                prediction_text = "not using"
            case 1:
                prediction_text = "using"
            case _:
                prediction_text = "unknown"

        # Render the rectangle onto the main frame.
        utils_general.render_detection_rectangle(frame_to_process, prediction_text, xyxy)
        cv2.imshow("Smartphone Usage Detection", frame_to_process)
    return num_people


if __name__ == "__main__":
    """ Utilities Initialization """
    # Device Support
    device = utils_general.get_device_support(torch)
    print(f"Device is using {device}.")

    # Detection Targets
    targets = utils_general.get_detection_targets()
    print(f"Successfully loaded target detection features.")

    # Frame Annotator Tools
    fa_pose_utils = FrameAnnotatorPoseUtils()
    fa_pose = FrameAnnotatorPose(general_utils=utils_general, annotator_utils=fa_pose_utils)
    print(f"Frame annotation utilities initialized.")

    """ Models """
    # YOLO Model
    YOLOv5s_model = YOLO('yolov5s.pt')
    print(f"YOLO model initialized: Running on {device}")

    # Initialize Mediapipe Posture Detection Model
    _, mp_pose = fa_pose_utils.init_mp()
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    print(f"Mediapipe pose detection model initialized.")

    # Self-trained Classification Model
    # TODO: Increase Model Stability & Accuracy
    with open("./data/models/posture_classify.pkl", "rb") as f:
        stc_model = pickle.load(f)
    with open("./data/models/posture_classify_scaler.pkl", "rb") as fs:
        stc_model_scaler = pickle.load(fs)
    print(f"Self-Trained pedestrian classification model initialized.")

    """ Video """
    cap = utils_general.init_video_capture()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        num_people = process_one_frame(
            frame,
            stc_model_and_scaler=[stc_model, stc_model_scaler],
            mp_pose_model=pose,
            YOLO_model=YOLOv5s_model,
            device=device
        )

        if utils_general.break_loop():
            break
