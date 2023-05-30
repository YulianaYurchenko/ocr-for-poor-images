import cv2
import os
import numpy as np


def perspective_transform(input_dir: str, filename: str, width: int, height: int, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, x4: int, y4: int):
    standart_path = os.path.normpath(input_dir)
    for root, dirs, files in os.walk(standart_path):
        for file in files:
            if file == filename:
                image = cv2.imread(filename)

    pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    result_image = cv2.warpPerspective(image, matrix, (width, height))

    output_dir = os.path.join(standart_path, 'perspective_transformed_images')

    output_path = os.path.join(output_dir,os.path.splitext(filename)[0] + "_perspective_transformed.jpg")

    cv2.imwrite(output_path, result_image)
