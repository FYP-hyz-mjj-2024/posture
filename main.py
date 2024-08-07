import utils

targets = [
    [("Left_shoulder", "Left_wrist"), "Left_elbow"],
    [("Right_shoulder", "Right_wrist"), "Right_elbow"],
    [("Left_hip", "Left_elbow"), "Left_shoulder"],
    [("Right_hip", "Right_elbow"), "Right_shoulder"],
]


def process_one_frame(frame, model, mp_drawing, connections, window_name="Untitled"):
    """
    Process one frame (one image).
    :param frame: The frame (or picture).
    :param model: The mediapipe posture detection model.
    :param mp_drawing: The mediapipe drawing tool acquired by mp.solutions.drawing_utils.
    :param connections: The frozenset that determines which landmarks are connected.
    :param window_name: The name of the opencv window.
    :return: The detected key angles.
    """

    """ Get Detection Results """
    results = utils.get_detection_results(frame, model)

    """ Extract Landmarks """
    key_coord_angles = None

    try:
        landmarks = results.pose_landmarks.landmark
        key_coord_angles = utils.gather_angles(landmarks, targets)
        utils.render_angles(frame, key_coord_angles)
    except Exception as e:
        print(f"Target is not in detection range. Error:{e}")

    """ Render Results """
    utils.render_results(frame, mp_drawing, results, connections, window_name)

    return key_coord_angles


def demo():
    # Initialize Drawing Tools and Detection Model.
    mp_drawing, mp_pose = utils.init_mp()

    # Setup mediapipe Instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # Initialize Media Source
        cap = utils.init_capture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            # Process One Frame
            data = process_one_frame(frame, pose, mp_drawing, mp_pose.POSE_CONNECTIONS, "Detections")

            # Save data if needed.
            utils.process_data(data)

            if utils.break_loop(show_preview=False):
                break

        cap.release()


if __name__ == "__main__":
    demo()
