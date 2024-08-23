import cv2
import torch
from PIL import Image
from pathlib import Path
import numpy as np


def get_static_image(path):
    img = Image.open(path)
    return img


def crop_pedestrians(img_file):
    """
    Input an image with multiple-pedestrians, use YOLOv5s to extract sub-images
    of individual pedestrians.
    :param img_file: An ImageFile object.
    :return: List of sub images.
    """

    # TODO: Local Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Extracts all the 4-d tuples corresponding to class 'person'
    results_df = model(img_file).pandas().xyxy[0]
    pedestrians = results_df[results_df['name'] == 'person']

    # Store cropped images
    cropped_images = []

    for idx, row in pedestrians.iterrows():
        cropped_img = img.crop(tuple(int(row[name]) for name in ['xmin', 'ymin', 'xmax', 'ymax']))
        cropped_images.append(np.array(cropped_img))

    return cropped_images


img = get_static_image("./data/_test/test_img.png")
cropped_pedestrians = crop_pedestrians(img)

for idx, cropped_img in enumerate(cropped_pedestrians):
    cv2.imwrite(f"./data/_test/cropped/{idx}.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))