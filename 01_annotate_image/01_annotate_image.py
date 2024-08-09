"""
01. Annotate image.
@author: Huang Yanzhen, Mai Jiajun
This file is used to annotate images. By annotate, we mean to extract target values from images using posture detection.
The target values are numerical to fit into further model training, e.g., the angle of the elbow, etc.
We don't want extra information from the images to prevent over-fitting.
"""

import os
import threading
from abc import ABC, abstractmethod
import json

# Local
import utils
from utils import FrameAnnotatorPoseUtils, FrameAnnotatorFaceUtils


class FrameAnnotator(ABC):

    def __init__(self, general_utils, annotator_utils):
        self.general_utils = general_utils
        self.annotator_utils = annotator_utils

    @abstractmethod
    def process_one_frame(
            self,
            frame,
            targets,
            model,
            mp_drawing,
            connections,
            window_name=None,
            window_shape=None,
            styles=None
    ):
        """
        Extract intended key features from a frame, and render the frame with the given model.
        :param frame: A video frame or a picture frame.
        :param model: The mediapipe detection model.
        :param targets: The intended detection targets.
        :param mp_drawing: The mediapipe drawing tool acquired by mp.solutions.drawing_utils.
        :param connections: The frozenset(s) that determines which landmarks are connected.
        :param window_name: The name of the opencv window. If not specified, the window will not be shown.
        :param window_shape: The size of the window. If not specified, defaults to config value.
        :param styles: The mediapipe drawing styles.
        :return: The detected values. If window_name is specified, a window may appear.
        """
        pass

    @abstractmethod
    def annotate_one_image(self, source_file_path, des_file_path, targets):
        """
        Extract intended key features from a frame, and use it to annotate an image.
        The results will be written into the destination file path.
        :param source_file_path: Path to the original image file, with .png extension.
        :param des_file_path: Path to the destination .json file where the data is written.
        :param targets: The intended detection targets.
        :return: None.
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
    def demo(self, cap, targets):
        """
        Starts a webcam demo.
        :param cap: An open-cv video capture object.
        :param targets: The intended detection targets.
        """
        pass


class FrameAnnotatorPose(FrameAnnotator):

    def process_one_frame(
            self,
            frame,
            targets,
            model,
            mp_drawing,
            connections,
            window_name=None,
            window_shape=None,
            styles=None
    ):
        # Get detection Results
        pose_results = self.annotator_utils.get_detection_results(frame, model)

        # Extract key angles with coordinates.
        key_coord_angles = None

        try:
            landmarks = pose_results.pose_landmarks.landmark
            key_coord_angles = self.annotator_utils.gather_angles(landmarks, targets)
            self.annotator_utils.render_angles(frame, key_coord_angles, window_shape)
        except Exception as e:
            print(f"Target is not in detection range. Error:{e}")

        # Render Results
        self.annotator_utils.render_results(
            frame,
            mp_drawing,
            pose_results,
            connections=connections,
            window_name=window_name,
            styles=styles
        )

        if key_coord_angles is not None:
            # Drop coordinates to save space. Coordinate is used only to draw on the canvas.
            for key_coord_angle in key_coord_angles:
                key_coord_angle.pop("coord")
            key_coord_angles = json.dumps(key_coord_angles, indent=4)
        return key_coord_angles

    def annotate_one_image(self, source_file_path, des_file_path, targets):

        # Initialize Drawing Tools and Detection Model.
        mp_drawing, mp_pose = self.annotator_utils.init_mp()

        # Initialize Media Source
        frame, frame_shape = self.general_utils.init_image_capture(source_file_path)

        # Setup mediapipe Instance
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Process One Frame
            data = self.process_one_frame(
                frame,
                targets,
                pose,
                mp_drawing,
                mp_pose.POSE_CONNECTIONS,
                window_name="Annotation",
                window_shape=frame_shape
            )

            # Save data if needed.
            self.general_utils.process_data(data, path=des_file_path)

    def batch_annotate_images(self, source_dir_path, des_dir_path, targets):

        for root, _, files in os.walk(source_dir_path):
            for file_name in files:
                if not file_name.endswith(".png"):
                    continue
                file_path = os.path.join(root, file_name)
                self.annotate_one_image(
                    file_path,
                    os.path.join(des_dir_path, file_name.replace(".png", ".json")),
                    targets
                )

                # TODO: This is weird. Need to fix.
                if self.general_utils.break_loop(show_preview=True):
                    continue

    def demo(self, cap, targets):
        print("Starting pose demo...")

        # Initialize Drawing Tools and Detection Model.
        mp_drawing, mp_pose = self.annotator_utils.init_mp()

        # Initialize Model
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        while cap.isOpened():
            ret, frame = cap.read()

            # Process One Frame
            data = self.process_one_frame(
                frame,
                targets,
                model=pose,
                mp_drawing=mp_drawing,
                connections=mp_pose.POSE_CONNECTIONS,
                window_name="Pose Estimation",
                window_shape=None,
                styles=None
            )

            if self.general_utils.break_loop(show_preview=False):
                break

        cap.release()


class FrameAnnotatorFace(FrameAnnotator):

    def process_one_frame(self, frame, targets, model, mp_drawing, connections, window_name="Untitled", window_shape=None, styles=None):
        """

        """
        """Get Detection Results"""
        face_results = self.annotator_utils.get_detection_results(frame, model)

        """Render Results"""
        self.annotator_utils.render_results(
            frame,
            mp_drawing,
            face_results,
            connections=connections,
            window_name=window_name,
            styles=styles
        )

    def annotate_one_image(self, source_file_path, des_file_path, targets):
        pass

    def batch_annotate_images(self, source_dir_path, des_dir_path, targets):
        pass

    def demo(self, cap, targets):
        print("Starting face demo...")

        # Initialize Drawing Tools and Detection Model.
        mp_drawing, mp_face_mesh, mp_drawing_styles = self.annotator_utils.init_mp()

        # Initialize Model
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        while cap.isOpened():
            ret, frame = cap.read()

            # Process One Frame
            self.process_one_frame(
                frame,
                targets,
                model=face_mesh,
                mp_drawing=mp_drawing,
                connections=[
                    mp_face_mesh.FACEMESH_TESSELATION,
                    mp_face_mesh.FACEMESH_CONTOURS,
                    mp_face_mesh.FACEMESH_IRISES
                ],
                window_name="Face Estimation",
                window_shape=None,
                styles=mp_drawing_styles
            )

            if self.general_utils.break_loop(show_preview=False):
                break

        cap.release()


def demo(funcs, func_targets):
    cap = utils.init_video_capture(0)
    threads = [threading.Thread(target=func, args=[cap, target]) for func, target in zip(funcs, func_targets)]
    try:
        print("Starting...")
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        print("Ended.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    face_targets = []
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
    ]

    # Initialize Utilities
    pfa_utils = FrameAnnotatorPoseUtils()
    ffa_utils = FrameAnnotatorFaceUtils()

    # Inject Utilities to Annotator
    fa_pose = FrameAnnotatorPose(general_utils=utils, annotator_utils=pfa_utils)
    fa_face = FrameAnnotatorFace(general_utils=utils, annotator_utils=ffa_utils)

    # demo([
    #     fa_pose.demo,
    #     fa_face.demo
    # ], [
    #     pose_targets,
    #     face_targets
    # ])

    fa_pose.batch_annotate_images(
        source_dir_path="../data/train/img/using",
        des_dir_path="../data/train/angles/using",
        targets=pose_targets)

    fa_pose.batch_annotate_images(
        source_dir_path="../data/train/img/not_using",
        des_dir_path="../data/train/angles/not_using",
        targets=pose_targets)

