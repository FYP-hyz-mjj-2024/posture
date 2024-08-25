import cv2
import torch


def crop_pedestrians(img_matrix):
    """
    Input an image with multiple-pedestrians, use YOLOv5s to extract sub-images
    of individual pedestrians.
    :param img_matrix: An ImageFile object.
    :return: List of sub images.
    """

    # TODO: Local Model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Image: BGR -> RGB
    img_rgb = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)

    # Extracts all the 4-d tuples corresponding to class 'person'
    results_df = model(img_matrix).pandas().xyxy[0]
    pedestrians = results_df[results_df['name'] == 'person']

    # Store cropped images
    cropped_images = []

    for idx, row in pedestrians.iterrows():
        xmin, ymin, xmax, ymax = tuple(int(row[name]) for name in ['xmin', 'ymin', 'xmax', 'ymax'])
        cropped_img = img_rgb[ymin:ymax, xmin:xmax]
        cropped_images.append(cropped_img)

    return cropped_images


if __name__ == "__main__":
    img = cv2.imread("./data/_test/test_img.png")
    cropped_pedestrians = crop_pedestrians(img)

    for idx, cropped_img in enumerate(cropped_pedestrians):
        cv2.imwrite(f"./data/_test/cropped/{idx}.png", cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))