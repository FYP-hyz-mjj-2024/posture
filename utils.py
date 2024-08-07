import numpy as np
import cv2

# Local
import config


def get_detection_results(frame, model):
    """
    Get the detection results for this frame with the given model.
    :param frame: An image frame from any source acquired from opencv-python.
    :param model: A mediapipe pose detection model acquired with mp.solutions.pose.
    :return: The original image frame.
    """
    # Re-color image: BGR (opencv preferred) -> RGB (mediapipe preferred)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Pose detection with model
    results = model.process(image)

    return results


def get_landmark_coords(landmarks, name):
    """
    Get the coordinates of the landmark with the given name.
    :param landmarks: Should acquire from results.pose_landmarks.landmark
    :param name: Name of the landmark
    :return: The 3-d coordinates of the landmark.
    """
    if not landmarks:
        return None

    landmark = landmarks[config.lm[name]]
    return [landmark.x, landmark.y, landmark.z]


def calc_angle(edge_points, mid_point):
    """
    Calculate the angle based on the given edge points and middle point.
    :param edge_points: The edge points of the angle.
    :param mid_point: The middle point of the angle, where the angle will be computed.
    :return: The degree value of the angle.
    """
    # Left, Right
    p1, p2 = [np.array(pt) for pt in edge_points]

    # Mid
    m = np.array(mid_point)

    # Angle
    radians = np.arctan2(p2[1]-m[1], p2[0]-m[0]) - np.arctan2(p1[1]-m[1], p1[0]-m[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle

    return angle


def calc_angle_lm(landmarks, edge_lm_names, mid_lm_name):
    """
    Calculate the angle based on the given edge landmarks and the middle landmark.
    The landmarks are specified with the standard name.
    :param landmarks: All the landmarks detected with the model.
    :param edge_lm_names: Names of the intended edge landmarks.
    :param mid_lm_name: Names of the intended middle landmarks.
    :return: The degree value of the angle.
    """
    n1, n2 = edge_lm_names
    nm = mid_lm_name
    return calc_angle(
        [get_landmark_coords(landmarks, n1), get_landmark_coords(landmarks, n2)],
        get_landmark_coords(landmarks, nm)
    )


def gather_angles(landmarks, targets):
    """
    Gather the angles based on the given targets.
    :param landmarks: All the landmarks detected with the model.
    :param targets: The targets to be gathered.
    :return: The gathered angles.
    """

    key_coord_angles = []

    for i, target in enumerate(targets):
        try:
            edge_lm_names, md_lm_name = target
            angle = calc_angle_lm(landmarks, edge_lm_names, md_lm_name)
            key_coord = get_landmark_coords(landmarks, md_lm_name)
            key_coord_angles.append({"coord": key_coord, "angle":angle})
        except Exception as e:
            print(e)
            continue

    return key_coord_angles


def render_angles(frame, landmarks, key_coord_angles):
    """
    Render the key coordinates based on the given targets.
    :param frame:
    :param landmarks:
    :param key_coord_angles:
    :return:
    """
    for key_coord_angle in key_coord_angles:
        cv2.putText(
            frame,
            str(round(key_coord_angle["angle"], 2)),
            tuple(np.multiply(key_coord_angle["coord"][:2], [640, 480]).astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
        )
