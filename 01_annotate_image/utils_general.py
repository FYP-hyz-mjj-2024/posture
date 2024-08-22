import cv2
import config


def init_image_capture(file_path):
    """
    Initialize an image capture source from a file.
    :param file_path: The path to the capture source file.
    :return: An image, the size of the image in width-height format.
    """
    image = cv2.imread(file_path)
    height, width, _ = image.shape
    return image, [width, height]


def init_video_capture(code=0):
    """
    Initialize a video capture source.
    :param code: The capture source.
    :return:
    """
    cap = cv2.VideoCapture(code)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.capture_shape[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.capture_shape[1])
    return cap


def break_loop(show_preview=False):
    """
    To determine whether to break the while-loop.
    :param show_preview: Whether is in batch mode.
    :return: The decision to terminate the while-loop or not.
    """
    return cv2.waitKey(int(not show_preview)) == 27


def process_data(data, path=None):
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
