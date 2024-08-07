import cv2
import utils

targets = [
    [("Left_shoulder", "Left_wrist"), "Left_elbow"],
    [("Right_shoulder", "Right_wrist"), "Right_elbow"],
    [("Left_hip", "Left_elbow"), "Left_shoulder"],
    [("Right_hip", "Right_elbow"), "Right_shoulder"],
]


def main():
    # Initialize Drawing Tools and Detection Model.
    mp_drawing, mp_pose = utils.init_mp()

    # Setup mediapipe Instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        cap = utils.init_capture(0)

        while cap.isOpened():
            ret, frame = cap.read()

            """ Get Detection Results """
            results = utils.get_detection_results(frame, pose)

            """ Extract Landmarks """
            try:
                landmarks = results.pose_landmarks.landmark
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


if __name__ == "__main__":
    main()
