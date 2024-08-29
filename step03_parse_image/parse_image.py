import cv2
from ultralytics import YOLO


def get_pedestrians_xyxy_list(img_matrix_rgb, YOLO_model_u, device):
    """
    Using YOLO object detection model, return a list of pedestrian coordinates.
    :param img_matrix_rgb: An image matrix from cv2.imread() that has been converted to RGB from BGR.
    :param YOLO_model_u: The YOLO model extracted from ultralytics
    """
    result = YOLO_model_u(
        img_matrix_rgb,
        classes=[0],
        device=device,
        conf=0.65,
        half=True
    )[0]
    bboxes = result.boxes.xyxy.cpu().numpy().astype("int")
    classes = result.boxes.cls.cpu().numpy().astype("int")

    xyxy_list = [box.tolist() for box, class_ in zip(bboxes, classes) if YOLO_model_u.names[class_] == "person"]
    return xyxy_list


def crop_pedestrians(img_matrix, model, device='cpu'):
    """
    Input an image with multiple-pedestrians, use YOLOv5s to extract sub-images
    of individual pedestrians.
    :param img_matrix: An ImageFile object.
    :return: List of sub images.
    """

    # Image: BGR -> RGB
    img_rgb = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)

    # Extracts all the 4-d tuples corresponding to class 'person'
    xyxy_list = get_pedestrians_xyxy_list(img_rgb, YOLO_model_u=model, device=device)

    # Store cropped images
    cropped_images = []

    for xyxy in xyxy_list:
        xmin, ymin, xmax, ymax = xyxy
        cropped_images.append(img_rgb[ymin:ymax, xmin:xmax])

    return cropped_images, xyxy_list


if __name__ == "__main__":
    YOLO_u = YOLO("yolov5s.pt")
    img = cv2.imread("./data/_test/test_img.png")
    cropped_pedestrians, _ = crop_pedestrians(img, YOLO_u)

    for idx, cropped_img in enumerate(cropped_pedestrians):
        cv2.imwrite(f"./data/_test/cropped/{idx}.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))

