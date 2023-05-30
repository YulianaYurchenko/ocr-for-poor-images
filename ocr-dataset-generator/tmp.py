import json
from pathlib import Path
from time import time
from tqdm import tqdm

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from skimage.measure import block_reduce

from utils import bfs, dfs
from src.transformations import rotate_and_cut_off, make_image_horizontal, cut_image_into_text_lines


with open('../DegradedImages/___train/degraded_images_data.json', 'r') as f:
    data = json.load(f)


gt = []
pred = []
pred_old = []

read_time = 0
morph_time = 0
block_reduce_time = 0
bfs_time = 0
linreg_time = 0
# rotate_time = 0
# cut_time = 0

i = 0
for img_name, img_data in tqdm(data.items()):
    if img_name[-4:] != '.png':
        continue

    i += 1
    if i == 1000:
        break

    t = time()
    image = cv.imread(str(Path('../DegradedImages/___train') / img_name), cv.IMREAD_GRAYSCALE)
    read_time += time() - t

    # a = np.mean(np.abs(np.diff(img, axis=1)), axis=1)

    # plt.figure(figsize=(15, 7))
    # plt.barh(np.arange(len(a)), a)
    # plt.grid()
    # plt.show()

    t = time()
    img = (image < np.mean(image) - np.std(image)).astype(float)
    img = cv.dilate(img, np.ones((1, img.shape[1] // 10)))
    img = cv.erode(img, np.ones((img.shape[0] // 40, img.shape[1] // 40)))
    morph_time += time() - t

    # img = np.pad(img, ((1, 1), (1, 1)))
    # d = np.diff(img.T, axis=1, prepend=0)
    # l = np.where(d == 1)
    # r = np.where(d == -1)
    # assert np.all(l[0] == r[0])
    #
    # group_indices = np.unique(r[0], return_index=True)[1][1:]  # r[0] should be sorted

    # np.split(r[1], group_idx) - np.split(l[1], group_idx)
    # cv.imshow('img', img)
    # cv.waitKey(0)
    img = img.astype(bool)

    t = time()
    img = block_reduce(img, (3, 3), np.max)
    block_reduce_time += time() - t

    # lines = np.where(img == 1)
    # cv.imshow('img', img)
    # cv.waitKey(0)
    h, w = img.shape
    used = np.zeros((h, w), dtype=bool)
    used_list = used.tolist()
    img_list = img.tolist()

    n = 0
    s = 0
    # rs = 0

    for x, y in zip(*np.where(~used & img)):

        if not used_list[x][y]:
            prev_used = used

            t = time()
            bfs(img_list, x, y, used_list, h, w)
            bfs_time += time() - t
            # img[used_copy != used] -= np.random.uniform(0, 1)

            used = np.array(used_list)
            _x, _y = np.where(prev_used != used)

            if len(_x) > w * h // 100:
                n += 1
                t = time()
                res = linregress(_y, _x)
                linreg_time += time() - t
                s += res.slope
                # rs += len(_x)

    s /= n
    s = np.arctan(s)

    gt.append(img_data['angle'])
    pred.append(-np.degrees(s))
    # print(n, -np.degrees(s))

    # t = time()
    # hor_image, a = make_image_horizontal(image, 10)
    # rotate_time += time() - t

    # pred_old.append(-a)
    #
    # t = time()
    # f = cut_image_into_text_lines(image)
    # cut_time += time() - t

    # if np.abs(img_data['angle']) < 0.5:
    #     print(np.abs(img_data['angle']))
    #     print(np.degrees(np.arctan(-s)))
    # cv.imshow('img', image)
    # cv.waitKey(0)

print(np.linalg.norm(np.array(gt) - np.array(pred)))
plt.figure(figsize=(15, 7))
plt.scatter(gt, pred)
# plt.scatter(gt, pred_old)
plt.plot([-10, 10], [-10, 10])
plt.grid()
plt.show()

print()
print(read_time)
print(morph_time)
print(block_reduce_time)
print(bfs_time)
print(linreg_time)
# print(rotate_time)
# print(cut_time)
