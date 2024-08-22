"""
01. Annotate image.
@author: Huang Yanzhen, Mai Jiajun
This file is used to annotate images. By annotate, we mean to extract target values from images using posture detection.
The target values are numerical to fit into further model training, e.g., the angle of the elbow, etc.
We don't want extra information from the images to prevent over-fitting.
"""

# Package
import os
import pickle
from abc import ABC, abstractmethod
import json
import cv2
import numpy as np

# Local
import utils_general as utils
from utils_annotation import FrameAnnotatorPoseUtils


class FrameAnnotator(ABC):

    def __init__(self, general_utils, annotator_utils):
        self.general_utils = general_utils
        self.annotator_utils = annotator_utils

    @abstractmethod
    def process_one_frame(self, frame, targets, model):
        """
        Extract intended key features from a frame, and render the frame with the given model.
        This function may or may not show the rendered frame, depending on whether invoker has
        specified the parameter window_name.
        :param frame: A video frame or a picture frame.
        :param model: The mediapipe detection model.
        :param targets: The intended detection targets.
        :return: The values of the detection targets, and the coordinates of all landmarks.
        """
        pass

    @abstractmethod
    def batch_annotate_images(self, source_dir_path, des_dir_path, targets):
        """
        Batch annotate images in a source directory. One image per file, one file per image.
        :param source_dir_path: Source directory that stores all the images to be annotated.
        :param des_dir_path: Destination directory that stores all the result files.
        :param targets: The intended detection targets for each annotation.
        :return: None.
        """
        pass

    @abstractmethod
    def demo(self, cap, targets, test_model=None):
        """
        Starts a webcam demo.
        :param cap: An open-cv video capture object.
        :param targets: The intended detection targets.
        """
        pass


class FrameAnnotatorPose(FrameAnnotator):

    def process_one_frame(self, frame, targets, model):
        # Get detection Results
        pose_results = self.annotator_utils.get_detection_results(frame, model)

        # Extract key angles with coordinates.
        key_coord_angles = None

        try:
            landmarks = pose_results.pose_landmarks.landmark
            key_coord_angles = self.annotator_utils.gather_angles(landmarks, targets)
        except Exception as e:
            print(f"Target is not in detection range. Error:{e}")

        """
        Type of key_coord_angles:
        [
            {"key":str, "coord":float, "angle":float},
            {"key":str, "coord":float, "angle":float},
            {"key":str, "coord":float, "angle":float},
            {"key":str, "coord":float, "angle":float},
            ...
        ]
        """
        return key_coord_angles, pose_results

    def batch_annotate_images(self, source_dir_path, des_dir_path, targets):

        # Initialize Drawing Tools and Detection Model.
        mp_drawing, mp_pose = self.annotator_utils.init_mp()

        for root, _, files in os.walk(source_dir_path):
            for file_name in files:

                # Annotate one image
                if not file_name.endswith(".png"):
                    continue

                # Specify src & des paths
                source_file_path = os.path.join(root, file_name)
                des_file_path = os.path.join(des_dir_path, file_name.replace(".png", ".json"))

                # Initialize Media Source
                frame, frame_shape = self.general_utils.init_image_capture(source_file_path)

                # Process One Frame
                with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
                    # Get the angle values and detected landmark coordinates.
                    key_coord_angles, pose_results = self.process_one_frame(frame, targets, model=pose)

                    # Render the detection
                    self.annotator_utils.render_angles(frame, key_coord_angles, frame_shape)
                    self.annotator_utils.render_results(
                        frame,
                        mp_drawing,
                        pose_results,
                        connections=mp_pose.POSE_CONNECTIONS,
                        window_name="Annotation",
                    )

                    # Save the key data into file.
                    if key_coord_angles is not None:
                        # Drop coordinates of key angles to save space.
                        # since they are only usable when rendering.
                        for key_coord_angle in key_coord_angles:
                            key_coord_angle.pop("coord")

                    key_coord_angles = json.dumps(key_coord_angles, indent=4)
                    # Save data if needed.
                    self.general_utils.process_data(key_coord_angles, path=des_file_path)

                # TODO: This is weird. Need to fix.
                if self.general_utils.break_loop(show_preview=True):
                    continue

    def demo(self, cap, targets, model_and_scaler=None):
        print("Starting pose demo...")

        # Initialize Drawing Tools and Detection Model.
        mp_drawing, mp_pose = self.annotator_utils.init_mp()

        # Initialize Model
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        while cap.isOpened():
            ret, frame = cap.read()

            # Get key angles, and pose detection results.
            key_coord_angles, pose_results = self.process_one_frame(frame, targets, model=pose)

            # Skip some initial frames
            if key_coord_angles is None:
                continue

            # Use the model to get the results
            if model_and_scaler is not None:
                # Prepare the key coordinate data
                this_model, scaler = model_and_scaler
                _numeric_data = np.array([kka['angle'] for kka in key_coord_angles]).reshape(1, -1)
                numeric_data = scaler.transform(_numeric_data)

                # Make the prediction
                prediction = this_model.predict(numeric_data)
                text = "not using" if prediction == 0 else "using"

                # Put the prediction results on the frame
                cv2.putText(
                    frame,
                    text,
                    (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

            # Render Results
            self.annotator_utils.render_angles(frame, key_coord_angles, window_shape=None)
            self.annotator_utils.render_results(
                frame,
                mp_drawing,
                pose_results,
                connections=mp_pose.POSE_CONNECTIONS,
                window_name="Test",
                styles=None
            )

            if self.general_utils.break_loop(show_preview=False):
                break

        cap.release()


if __name__ == "__main__":
    pose_targets = [
        # Arms
        [("Left_shoulder", "Left_wrist"), "Left_elbow"],
        [("Right_shoulder", "Right_wrist"), "Right_elbow"],
        [("Left_hip", "Left_elbow"), "Left_shoulder"],
        [("Right_hip", "Right_elbow"), "Right_shoulder"],

        # Face-Shoulder
        [("Right_shoulder", "Left_shoulder"), "Nose"],
        [("Right_eye_outer", "Nose"), "Right_shoulder"],
        [("Left_eye_outer", "Nose"), "Left_shoulder"],
        [("Right_eye", "Right_ear"), "Right_eye_outer"],
        [("Left_eye", "Left_ear"), "Left_eye_outer"],
    ]

    # Initialize Utilities
    fa_pose_utils = FrameAnnotatorPoseUtils()

    # Inject Utilities to Annotator
    fa_pose = FrameAnnotatorPose(general_utils=utils, annotator_utils=fa_pose_utils)

    """ 
    Model Prediction Demo 
    """
    with open("../data/models/posture_classify.pkl", "rb") as f:
        model = pickle.load(f)

    with open("../data/models/posture_classify_scaler.pkl", "rb") as fs:
        model_scaler = pickle.load(fs)

    cap = utils.init_video_capture(0)
    fa_pose.demo(cap, pose_targets, [model, model_scaler])

    """ 
    Image Annotation 
    """
    # fa_pose.batch_annotate_images(
    #     source_dir_path="../data/train/img/using",
    #     des_dir_path="../data/train/angles/using",
    #     targets=pose_targets)
    #
    # fa_pose.batch_annotate_images(
    #     source_dir_path="../data/train/img/not_using",
    #     des_dir_path="../data/train/angles/not_using",
    #     targets=pose_targets)

