import json
from pathlib import Path
from time import time
from tqdm import tqdm

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.measure import block_reduce

from utils import bfs
from src.transformations import rotate_and_cut_off
from _ import cut_image_into_text_lines_new


root_path = Path('../DegradedImages/data')
with open(root_path / 'degraded_images_data.json', 'r') as f:
    data = json.load(f)


gt = []
pred = []

i = 0
for img_name, img_data in tqdm(data.items()):
    if img_name[-4:] != '.png':
        continue

    i += 1

    image = cv.imread(str(root_path / img_name), cv.IMREAD_GRAYSCALE)

    img = (image < np.mean(image) - np.std(image)).astype(float)
    img = cv.dilate(img, np.ones((1, img.shape[1] // 10)))
    img = cv.erode(img, np.ones((img.shape[0] // 40, img.shape[1] // 40)))

    img = img.astype(bool)

    img = block_reduce(img, (3, 3), np.max)

    h, w = img.shape
    used = np.zeros((h, w), dtype=bool)
    used_list = used.tolist()
    img_list = img.tolist()

    slopes = []

    for x, y in zip(*np.where(~used & img)):

        if not used_list[x][y]:
            prev_used = used

            bfs(img_list, x, y, used_list, h, w)

            used = np.array(used_list)
            _x, _y = np.where(prev_used != used)

            if len(_x) > w * h // 100:
                res = linregress(_y, _x)
                slopes.append(res.slope)

    s = np.mean(slopes)
    s = np.arctan(s)

    gt.append(img_data['angle'])
    pred.append(-np.degrees(s))

    rot_image = rotate_and_cut_off(image, s, (image.shape[1] // 2, image.shape[0] // 2))

    single_line_images, p = cut_image_into_text_lines_new(rot_image)
    if p != img_data['n']:
        print(i)
    for k in single_line_images:
        cv.imshow('img', k)
        cv.waitKey(0)


print(np.linalg.norm(np.array(gt) - np.array(pred)))
plt.figure(figsize=(15, 7))
plt.scatter(gt, pred, alpha=0.3)
plt.plot([-10, 10], [-10, 10])
plt.grid()
plt.show()
