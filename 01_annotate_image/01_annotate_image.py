"""
01. Annotate image.
@author: Huang Yanzhen, Mai Jiajun
This file is used to annotate images. By annotate, we mean to extract target values from images using posture detection.
The target values are numerical to fit into further model training, e.g., the angle of the elbow, etc.
We don't want extra information from the images to prevent over-fitting.
"""

import os
import threading
import json

import utils
from utils import PoseFrameAnnotator, FaceMeshFrameAnnotator


pfa = PoseFrameAnnotator()
ffa = FaceMeshFrameAnnotator()

targets = [
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


def process_one_frame_pose(frame, model, mp_drawing, connections, window_name=None, window_shape=None, styles=None):
    """

    """
    """ Get Detection Results """
    pose_results = pfa.get_detection_results(frame, model)

    """ Extract Landmarks """
    key_coord_angles = None

    try:
        landmarks = pose_results.pose_landmarks.landmark
        key_coord_angles = pfa.gather_angles(landmarks, targets)
        pfa.render_angles(frame, key_coord_angles, window_shape)
    except Exception as e:
        print(f"Target is not in detection range. Error:{e}")

    """ Render Results """
    pfa.render_results(
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

def process_one_frame_face(frame, model, mp_drawing, connections, window_name="Untitled", window_shape=None, styles=None):
    """

    """
    """Get Detection Results"""
    face_results = ffa.get_detection_results(frame, model)

    """Render Results"""
    ffa.render_results(
        frame,
        mp_drawing,
        face_results,
        connections=connections,
        window_name=window_name,
        styles=styles
    )


def process_one_frame(frame, models, mp_drawing, connections, window_name="Untitled", window_shape=None, styles=None):
    """
    Process one frame (one image).
    :param frame: The frame (or picture).
    :param models: A list of mediapipe detection models.
    :param mp_drawing: The mediapipe drawing tool acquired by mp.solutions.drawing_utils.
    :param connections: A list of corresponding rendering connections according to the sequence of the model.
    :param window_name: The name of the opencv window.
    :param window_shape: The size of the window. If not specified, defaults to config value.
    :return: The detected key angles.
    """

    pose_model, face_model = models
    pose_connections, face_connections = connections

    """ Get Detection Results """
    pose_results = pfa.get_detection_results(frame, pose_model)
    face_results = ffa.get_detection_results(frame, face_model)

    """ Extract Landmarks """
    key_coord_angles = None

    try:
        landmarks = pose_results.pose_landmarks.landmark
        key_coord_angles = pfa.gather_angles(landmarks, targets)
        pfa.render_angles(frame, key_coord_angles, window_shape)
    except Exception as e:
        print(f"Target is not in detection range. Error:{e}")

    """ Render Results """
    pfa.render_results(
        frame,
        mp_drawing,
        pose_results,
        connections=pose_connections
    )

    ffa.render_results(
        frame,
        mp_drawing,
        face_results,
        connections=face_connections,
        window_name=window_name,
        styles=styles
    )

    if key_coord_angles is not None:
        # Drop coordinates to save space. Coordinate is used only to draw on the canvas.
        for key_coord_angle in key_coord_angles:
            key_coord_angle.pop("coord")
        key_coord_angles = json.dumps(key_coord_angles, indent=4)

    return key_coord_angles


def annotate_one_image(source_file_path, des_file_path):
    """
    Detect key angles of only one image and write the results to the corresponding path.
    :param source_file_path: The original image file, with .png extension.
    :param des_file_path: The destination file where the data is written.
    :return: None.
    """
    # Initialize Drawing Tools and Detection Model.
    mp_drawing, mp_pose = pfa.init_mp()

    # Initialize Media Source
    frame, frame_shape = utils.init_image_capture(source_file_path)

    # Setup mediapipe Instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        # Process One Frame
        data = process_one_frame(frame, pose, mp_drawing, mp_pose.POSE_CONNECTIONS,
                                 window_name="Annotation",
                                 window_shape=frame_shape)

        # Save data if needed.
        utils.process_data(data, path=des_file_path)


def batch_annotate_images(source_dir_path, des_dir_path):
    """
    Batch annotate images in a source directory. One image per file, one file per image.
    :param source_dir_path: Source directory that stores all the images to be annotated.
    :param des_dir_path: Destination directory that stores all the result files.
    :return: None.
    """
    for root, _, files in os.walk(source_dir_path):
        for file_name in files:
            if not file_name.endswith(".png"):
                continue
            file_path = os.path.join(root, file_name)
            annotate_one_image(file_path, os.path.join(des_dir_path, file_name.replace(".png", ".json")))

            if utils.break_loop(show_preview=True):
                continue


def demo_pose(cap):
    print("Starting pose demo...")

    # Initialize Drawing Tools and Detection Model.
    mp_drawing, mp_pose = pfa.init_mp()

    # Initialize Model
    pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()

        # Process One Frame
        data = process_one_frame_pose(
            frame,
            model=pose,
            mp_drawing=mp_drawing,
            connections=mp_pose.POSE_CONNECTIONS,
            window_name="Pose Estimation",
            window_shape=None,
            styles=None
        )

        if utils.break_loop(show_preview=False):
            break

    cap.release()


def demo_face(cap):
    print("Starting face demo...")

    # Initialize Drawing Tools and Detection Model.
    mp_drawing, mp_face_mesh, mp_drawing_styles = ffa.init_mp()

    # Initialize Model
    face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()

        # Process One Frame
        process_one_frame_face(
            frame,
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

        if utils.break_loop(show_preview=False):
            break

    cap.release()


def demo(funcs):
    cap = utils.init_video_capture(0)
    threads = [threading.Thread(target=func, args=[cap]) for func in funcs]
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
    demo([
        demo_pose,
        demo_face
    ])
