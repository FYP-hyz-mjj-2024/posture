# Shape of window.
capture_shape = [640, 480]

# Mediapipe options.
opt = {
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
    "show_frame_preview": False,
    "save_results": False,
    "save_path": "./saves/tmp/"
}

# Pose code according to mediapipe.
lm_pose = {
    "Nose": 0,
    "Left_eye_inner": 1,
    "Left_eye": 2,
    "Left_eye_outer": 3,
    "Right_eye_inner": 4,
    "Right_eye": 5,
    "Right_eye_outer": 6,
    "Left_ear": 7,
    "Right_ear": 8,
    "Mouth_left": 9,
    "Mouth_right": 10,
    "Left_shoulder": 11,
    "Right_shoulder": 12,
    "Left_elbow": 13,
    "Right_elbow": 14,
    "Left_wrist": 15,
    "Right_wrist": 16,
    "Left_pinky": 17,
    "Right_pinky": 18,
    "Left_index": 19,
    "Right_index": 20,
    "Left_thumb": 21,
    "Right_thumb": 22,
    "Left_hip":23,
    "Right_hip": 24,
    "Left_knee": 25,
    "Right_knee": 26,
    "Left_ankle": 27,
    "Right_ankle": 28,
    "Left_heel": 29,
    "Right_heel": 30,
    "Left_foot_index": 31,
    "Right_foot_index": 32,
}

# Detection Targets
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
    [("Left_shoulder", "Right_shoulder"), "Left_ear"],
    [("Left_shoulder", "Right_shoulder"), "Right_ear"],

    # Hand
    [("Left_elbow", "Left_pinky"), "Left_wrist"],
    [("Right_elbow", "Right_pinky"), "Right_wrist"],
    [("Left_index", "Left_pinky"), "Left_wrist"],
    [("Right_index", "Right_pinky"), "Right_wrist"],
]

# The server URL of the websocket destination of video stream pushing.
websocket_server_url = "ws://localhost:8080"