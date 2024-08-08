from abc import ABC, abstractmethod
import cv2
import numpy as np
import mediapipe as mp

import config


class FrameAnnotator(ABC):

    def init_image_capture(self, file_path):
        """
        Initialize an image capture source from a file.
        :param file_path: The path to the capture source file.
        :return: An image, the size of the image in width-height format.
        """
        image = cv2.imread(file_path)
        height, width, _ = image.shape
        return image, [width, height]

    def init_video_capture(self, code=0):
        """
        Initialize a video capture source.
        :param code: The capture source.
        :return:
        """
        cap = cv2.VideoCapture(code)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_shape[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_shape[1])
        return cap

    def calc_angle(self, edge_points, mid_point):
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
        radians = np.arctan2(p2[1] - m[1], p2[0] - m[0]) - np.arctan2(p1[1] - m[1], p1[0] - m[0])
        angle = np.abs(radians * 180.0 / np.pi)
        if angle > 180.0:
            angle = 360 - angle

        return angle

    def break_loop(self, show_preview=False):
        """
        To determine whether to break the while-loop.
        :param show_preview: Whether is in batch mode.
        :return: The decision to terminate the while-loop or not.
        """
        return cv2.waitKey(int(not show_preview)) == 27

    def process_data(self, data, path=None):
        """
        Process the retrieved angles.
        :param data: The angles.
        :param path: The save path of the file.
        :return: None.
        """
        if data is None or path is None:
            return

        with open(path, "w") as f:
            f.write(str(data))

    @abstractmethod
    def init_mp(self):
        pass

    @abstractmethod
    def get_detection_results(self, frame, model):
        pass

    @abstractmethod
    def get_landmark_coords(self, landmarks, name):
        pass

    @abstractmethod
    def render_results(self, frame, mp_drawing, results, connections, window_name):
        pass


class PoseFrameAnnotator(FrameAnnotator):
    def init_mp(self):
        """
        Initialize mediapipe essentials. Including
        - Mediapipe drawing tools.
        - Mediapipe posture detection model (pose).
        :return: Drawing Utilities, Pose Model
        """

        # Drawing Utilities
        mp_drawing = mp.solutions.drawing_utils

        # Pose Estimation Model
        mp_pose = mp.solutions.pose
        return mp_drawing, mp_pose

    def get_detection_results(self, frame, model):
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

    def get_landmark_coords(self, landmarks, name):
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

    def calc_angle(self, edge_points, mid_point):
        return super().calc_angle(edge_points, mid_point)

    def calc_angle_lm(self, landmarks, edge_lm_names, mid_lm_name):
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
        return self.calc_angle(
            [self.get_landmark_coords(landmarks, n1), self.get_landmark_coords(landmarks, n2)],
            self.get_landmark_coords(landmarks, nm)
        )

    def gather_angles(self, landmarks, targets):
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

                # Angle between the landmarks
                angle = self.calc_angle_lm(landmarks, edge_lm_names, md_lm_name)

                # Key Coordinate of the middle point for rendering purposes.
                key_coord = self.get_landmark_coords(landmarks, md_lm_name)

                # Name of this feature
                key = f"{edge_lm_names[0]}-{md_lm_name}-{edge_lm_names[1]}"

                key_coord_angles.append({"key": key, "coord": key_coord, "angle": angle})
            except Exception as e:
                print(e)
                continue

        return key_coord_angles

    def render_angles(self, frame, key_coord_angles, window_shape=None):
        """
        Render the key coordinates based on the given targets.
        :param frame: An image frame from any source acquired from opencv-python.
        :param key_coord_angles: A list of key-coordinate-angle tuple that records the key angles as features.
        :param window_shape: The shape of the window used to render the key angle values at the key coordinates.
        :return: None.
        """
        if window_shape is None:
            window_shape = config.capture_shape

        for key_coord_angle in key_coord_angles:
            cv2.putText(
                frame,
                str(round(key_coord_angle["angle"], 2)),
                tuple(np.multiply(key_coord_angle["coord"][:2], window_shape).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
            )

    def render_results(self, frame, mp_drawing, results, connections, window_name):
        """
        Render all the detection results.
        :param frame: An image frame from any source acquired from opencv-python.
        :param mp_drawing: Mediapipe drawing tools acquired by mp.solutions.drawing_utils.
        :param results: The detected results
        :param connections: A frozen set of connections that defines which landmarks are connected.
        :param window_name: The name of the opened window. Default to be "Untitled".
        :return: None
        """
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, connections,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow(window_name, frame)

    def process_data(self, data, path=None):
        return super().process_data(data, path)

    def break_loop(self, show_preview=False):
        return super().break_loop(show_preview)
