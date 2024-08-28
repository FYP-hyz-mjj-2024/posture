"""
01. Annotate image.
@author: Huang Yanzhen, Mai Jiajun
This file is used to annotate images. By annotate, we mean to extract target values from images using posture detection.
The target values are numerical to fit into further YOLO_model training, e.g., the angle of the elbow, etc.
We don't want extra information from the images to prevent over-fitting.
"""

# Package
import os
from abc import ABC, abstractmethod
import cv2
import numpy as np
import pandas as pd

# Local
if __name__ == "__main__":
    import utils_general as utils
    from utils_annotation import FrameAnnotatorPoseUtils
else:
    from . import utils_general as utils
    from .utils_annotation import FrameAnnotatorPoseUtils


class FrameAnnotator(ABC):

    def __init__(self, general_utils, annotator_utils):
        self.general_utils = general_utils
        self.annotator_utils = annotator_utils

    @abstractmethod
    def process_one_frame(self, frame, targets, model):
        """
        Extract intended key features from a frame, and render the frame with the given YOLO_model.
        This function may or may not show the rendered frame, depending on whether invoker has
        specified the parameter window_name.
        :param frame: A video frame or a picture frame.
        :param model: The mediapipe detection YOLO_model.
        :param targets: The intended detection targets.
        :return: The values of the detection targets, and the coordinates of all landmarks.
        """
        pass

    @abstractmethod
    def batch_annotate_images(self, source_dir_path, des_file, targets, labels):
        """
        Batch annotate images in a source directory. One image per file, one file per image.
        :param source_dir_path: Source directory that stores all the images to be annotated.
        :param des_file: File path to the destination csv file.
        :param targets: The intended detection targets for each annotation.
        :param labels: The label for this annotation batch.
        :return: None.
        """
        pass

    @abstractmethod
    def demo(self, cap, targets, model_and_scaler=None):
        """
        Starts a webcam demo.
        :param cap: An open-cv video capture object.
        :param targets: The intended detection targets.
        :param model_and_scaler: A tuple of YOLO_model-scaler object used to make predictions on the input data.
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

    def batch_annotate_images(self, source_dir, des_file, targets, label):

        # Initialize Drawing Tools and Detection Model.
        mp_drawing, mp_pose = self.annotator_utils.init_mp()

        # Initialize an empty dataframe.
        df_data = {}

        # Run through all the files.
        for root, _, files in os.walk(source_dir):
            for file_name in files:
                # Annotate one image
                if not file_name.endswith(".png"):
                    continue

                # Specify src path.
                source_file_path = os.path.join(root, file_name)

                # Get image.
                frame, frame_shape = self.general_utils.init_image_capture(source_file_path)

                # From this image, get the key angles.
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
                            # key_coord_angle["labels"] = 0 if labels == "not_using" else 1

                    # Save angle vector into dataframe.
                    for key_angle in key_coord_angles:
                        if key_angle["key"] not in df_data:
                            df_data[key_angle["key"]] = [key_angle["angle"]]
                        else:
                            df_data[key_angle["key"]].append(key_angle["angle"])

                # TODO: This is weird. Need to fix.
                if self.general_utils.break_loop(show_preview=True):
                    continue

        # Save the dataframe into csv.
        df = pd.DataFrame.from_dict(df_data)
        df['labels'] = label
        df.to_csv(des_file, index=False)
        print(df)

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

            # Use the YOLO_model to get the results
            if model_and_scaler is not None:
                # Prepare the key coordinate data
                this_model, scaler = model_and_scaler
                _numeric_data = np.array([kka['angle'] for kka in key_coord_angles]).reshape(1, -1)
                numeric_data = scaler.transform(_numeric_data)

                # Make the prediction
                prediction = this_model.predict(numeric_data)
                match prediction:
                    case 0:
                        text = "not using"
                    case 1:
                        text = "using"
                    case _:
                        text = "unknown"

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
    # Initialize Detection Targets
    pose_targets = utils.get_detection_targets()

    # Initialize Utilities
    fa_pose_utils = FrameAnnotatorPoseUtils()

    # Inject Utilities to Annotator
    fa_pose = FrameAnnotatorPose(general_utils=utils, annotator_utils=fa_pose_utils)

    """ 
    Image Annotation 
    """
    fa_pose.batch_annotate_images(
        source_dir="../data/train/img/using",
        des_file="../data/train/using.csv",
        targets=pose_targets,
        label=1)

    fa_pose.batch_annotate_images(
        source_dir="../data/train/img/not_using",
        des_file="../data/train/not_using.csv",
        targets=pose_targets,
        label=0)

