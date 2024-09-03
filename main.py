# Package
import base64
import functools
import websocket
import json
import time
import cv2
import pickle
import torch
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Local
import step01_annotate_image.utils_general as utils_general
from step01_annotate_image.annotate_image import FrameAnnotatorPose
from step01_annotate_image.utils_annotation import FrameAnnotatorPoseUtils
from step03_parse_image.parse_image import crop_pedestrians


def annotate_one_person(
        frame_to_process,
        stc_model_and_scaler,
        mp_pose_model,
        pedestrian_frame,
        xyxy) -> None:
    """
    Annotate a single person given in a single frame. Render the annotation results
    onto the entire frame.
    :param frame_to_process: The entire frame.
    :param stc_model_and_scaler: The self-trained classification model and scaler.
    :param pedestrian_frame: The subframe from the entire frame that contains the target person.
    :param xyxy: The coordinates of the subframe.
    :param mp_pose_model: The posture detection model.
    """

    # Extract self-trained classification model and scaler.
    stc_model, stc_model_scaler = stc_model_and_scaler

    # Get the key angle array from a subframe.
    pedestrian_frame = cv2.cvtColor(pedestrian_frame, cv2.COLOR_RGB2BGR)
    key_coord_angles, _ = fa_pose.process_one_frame(
        pedestrian_frame,
        targets=targets,
        model=mp_pose_model
    )

    if key_coord_angles is None:
        return

    _numeric_data = np.array([kka['angle'] for kka in key_coord_angles]).reshape(1, -1)

    # Feed the normalized angle array into the self-trained model, get prediction.
    numeric_data = stc_model_scaler.transform(_numeric_data)
    prediction_boolean = stc_model.predict(numeric_data)
    match prediction_boolean:
        case 0:
            prediction_text = "not using"
        case 1:
            prediction_text = "using"
        case _:
            prediction_text = "unknown"

    # Render the rectangle + predictions onto the main frame.
    utils_general.render_detection_rectangle(frame_to_process, prediction_text, xyxy)


def process_one_frame(
        frame_to_process,
        stc_model_and_scaler,
        mp_pose_model,
        YOLO_model=None,
        device='cpu'):
    """
    Take a single frame. Use YOLO model to locate all the pedestrians. Then, for each retrieved
    pedestrian, feed it into mediapipe posture detection model to extract landmarks. After that,
    calculate key features from the landmarks, and feed to the self-trained classification model.
    The self-trained classification model will return the prediction results, which will be then
    rendered on the frame.
    :param frame_to_process: The video frame retrieved from cv2.VideoCapture().read()
    :param stc_model_and_scaler: A tuple/array of the self-trained classification model and scaler.
    :param mp_pose_model: The mediapipe posture detection model.
    :param YOLO_model: The YOLO model for pedestrian location & image cropping.
    :param device: GPU support.
    :return: Annotated Frame, [Number of People, YOLO consumption Time, Classification Consumption Time]
    """
    # Crop out pedestrians
    start_time_YOLO = time.time()
    pedestrian_frames, xyxy_sets = crop_pedestrians(frame_to_process, model=YOLO_model, device=device)
    time_YOLO = time.time() - start_time_YOLO

    if (pedestrian_frames is None) or (xyxy_sets is None):
        # cv2.imshow("Smartphone Usage Detection", frame_to_process)
        return frame_to_process, [0, time_YOLO, 0]

    # Number of people
    num_people = len(pedestrian_frames)

    if num_people <= 0:
        # cv2.imshow("Smartphone Usage Detection", frame_to_process)
        return frame_to_process, [0, time_YOLO, 0]

    # Process each person (subframe)
    # Use lambda for-loops for better performance
    start_time_classification = time.time()
    [annotate_one_person(frame_to_process, stc_model_and_scaler, mp_pose_model, pedestrian_frame, xyxy)
     for pedestrian_frame, xyxy in zip(pedestrian_frames, xyxy_sets)]
    time_classification = time.time() - start_time_classification

    # cv2.imshow("Smartphone Usage Detection", frame_to_process)

    return frame_to_process, [num_people, time_YOLO, time_classification]


def plot_performance_report(arrays, labels, config) -> None:
    """
    Plot the performance report.
    :param arrays: The performance indications.
    :param labels: The labels of each curve.
    :param config: Plot configurations.
    """
    if not all(len(array) == len(arrays[0]) for array in arrays):
        raise ValueError("All arrays must be the same length.")

    plt.figure(figsize=(10, 6))
    iterations = [i for i in range(len(arrays[0]))]

    for arr, label in zip(arrays, labels):
        mean = np.mean(arr)
        plt.plot(iterations, arr, label=f"{label}")
        plt.plot(iterations, [mean for _ in range(len(arr))], linestyle='--', label=f"{label} - Mean={mean:.2f}")

    plt.title(config['title'])
    plt.xlabel(config['x_name'])
    plt.ylabel(config['y_name'])
    plt.legend()
    plt.grid(True)
    plt.show()


def yield_video_feed(frame_to_yield, mode='local', title="", ws=None) -> None:
    """
    Yield the video frame. Either using local mode, which will invoke an
    opencv imshow window, or use the HTTP Streaming to the server.
    :param frame_to_yield: The video frame.
    :param mode: Yielding mode, either be 'local' or 'remote'.
    :param title: The title of the local window.
    :param ws: The websocket object initialized with server_url.
    """
    if mode == 'local':
        cv2.imshow(title, frame_to_yield)
    elif mode == 'remote':
        if ws is None:
            raise ValueError("WebSocket object is not initialized.")
        # JPEG encode, convert to bytes
        _, jpeg_encoded = cv2.imencode('.jpg', frame_to_yield)
        jpeg_bytes = jpeg_encoded.tobytes()
        jpeg_base64 = base64.b64encode(jpeg_bytes).decode('utf-8')

        # Send request
        ws.send(json.dumps({'message':jpeg_base64}))
    else:
        raise ValueError("Video yielding mode should be either 'local' or 'remote'.")


def init_websocket(server_url) -> websocket.WebSocket | None:
    """
    Initialize a websocket object using the url of the server.
    :param server_url: The url of the server.
    """
    try:
        ws = websocket.WebSocket()
        ws.connect(server_url)
        return ws
    except ConnectionRefusedError as e:
        print(f"Connection to WebSocked Failed. The server might be closed. Error: {e}\n"
              f"If you are using local mode, you can ignore this error.")
        return None


if __name__ == "__main__":
    """ Utilities Initialization """
    # Device Support
    device = utils_general.get_device_support(torch)
    print(f"Device is using {device}.")

    # Detection Targets
    targets = utils_general.get_detection_targets()
    print(f"Successfully loaded target detection features.")

    # Frame Annotator Tools
    fa_pose_utils = FrameAnnotatorPoseUtils()
    fa_pose = FrameAnnotatorPose(general_utils=utils_general, annotator_utils=fa_pose_utils)
    print(f"Frame annotation utilities initialized.")

    """ Models """
    # YOLO Model
    YOLOv5s_model = YOLO('yolov5s.pt')
    print(f"YOLO model initialized: Running on {device}")

    # Initialize Mediapipe Posture Detection Model
    _, mp_pose = fa_pose_utils.init_mp()
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
    print(f"Mediapipe pose detection model initialized.")

    # Self-trained Classification Model
    # TODO: Increase Model Stability & Accuracy
    with open("./data/models/posture_classify.pkl", "rb") as f:
        stc_model = pickle.load(f)
    with open("./data/models/posture_classify_scaler.pkl", "rb") as fs:
        stc_model_scaler = pickle.load(fs)
    print(f"Self-Trained pedestrian classification model initialized.")

    """ Video """
    cap = utils_general.init_video_capture(0)

    # Performance Analysis
    report = {
        'Total Time': [],
        'YOLO Time': [],
        'Classification Time': []
    }

    # Initialize Web Socket
    server_url = utils_general.get_websocket_server_url()
    ws = init_websocket(server_url)

    # Video Frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        start_time = time.time()
        processed_frame, [num_people, time_YOLO, time_classification] = process_one_frame(
            frame,
            stc_model_and_scaler=[stc_model, stc_model_scaler],
            mp_pose_model=pose,
            YOLO_model=YOLOv5s_model,
            device=device
        )
        report['Total Time'].append(time.time() - start_time)
        report['YOLO Time'].append(time_YOLO)
        report['Classification Time'].append(time_classification)

        # cv2.imshow("Smartphone Usage Detection", processed_frame)
        yield_video_feed(processed_frame, mode='local', title="Smartphone Usage Detection")
        # yield_video_feed(processed_frame, mode='remote', ws=ws)

        if utils_general.break_loop():
            break

    # Remove performance data of the first frame.
    # Initialization takes significant time, bad for mean value.
    report = {key: [0] + value[1:] for key, value in report.items()}

    # Performance Report.
    plot_performance_report(
        list(report.values()),
        report.keys(),
        {
            'title': 'Frame Computation Time',
            'x_name': 'Frame Number',
            'y_name': 'Time (s)'
        }
    )

