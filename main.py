import cv2
import mediapipe as mp
import numpy as np

import utils

# Drawing Utilities
mp_drawing = mp.solutions.drawing_utils

# Pose Estimation Model
mp_pose = mp.solutions.pose

# Setup mediapipe Instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()

        """ Get Detection Results """
        results = utils.get_detection_results(frame, pose)

        """ Extract Landmarks """
        try:
            landmarks = results.pose_landmarks.landmark
            targets = [
                [("Left_shoulder", "Left_wrist"), "Left_elbow"],
                [("Right_shoulder", "Right_wrist"), "Right_elbow"],
                [("Left_hip", "Left_elbow"), "Left_shoulder"],
                [("Right_hip", "Right_elbow"), "Right_shoulder"],
            ]

            key_coord_angles = utils.gather_angles(landmarks, targets)
            utils.render_angles(frame, landmarks, key_coord_angles)
        except Exception as e:
            print(e)
            pass

        """ Render Results """
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('Mediapipe Feed', frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()

