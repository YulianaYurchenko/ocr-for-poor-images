from typing import List

import numpy as np
import cv2 as cv
from math import sqrt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.stats import linregress
from skimage.measure import block_reduce
from scipy.signal import find_peaks

# Desired image dimensions
IMG_WIDTH = 256
IMG_HEIGHT = 32

# All characters that can happen in labels
characters = ['\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']


# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


def encode_single_image(image):
    # 1. Convert grayscale image to 3-dimensional tensor
    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    # 2. Convert to float32 in [0, 1] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 3. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    image = tf.transpose(image, perm=[1, 0, 2])
    return image


# A utility function to decode the output of the network
def decode_batch_predictions(pred, max_label_len):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_label_len]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res + 1)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


def take_clear_image_text(data, clear_image_name):
    res = ''
    clear_image_data = data[clear_image_name]

    for word_data in clear_image_data['words']:
        word = str(word_data['word'])
        if word.find('\n') == -1:
            res += word + ' '
        else:
            res += word
    res = res[:-1]
    res += '\f'
    return res


def rotate_and_cut_off(image: np.ndarray, angle: float, center: (int, int)) -> np.ndarray:
    """
        Function rotates an image and cuts off borders

        Args:
            image (np.array): tensor representing an image.
            angle (float): the angle of rotation is measured in degrees.
            center ((int, int)): center of rotation.
    """
    height, width = image.shape[:2]
    x, y = center

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(-height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t - width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    rotated = cv.warpAffine(image, M, (new_width, new_height))
    return rotated


def cut_horizontal_image(image: np.ndarray) -> List[np.ndarray]:

    x = np.mean(1 - image, axis=1) / np.mean(1 - image)
    kernel = image.shape[0] // 20
    x = np.convolve(x, np.ones(kernel) / kernel, mode='same')

    peaks, _ = find_peaks(x, height=0.75, prominence=0.3, width=image.shape[0] // 20)
    cut_indices = (peaks[1:] + peaks[:-1]) // 2
    cut_indices = np.insert(cut_indices, 0, 0)
    cut_indices = np.append(cut_indices, image.shape[0])

    single_line_images = []
    for i1, i2 in zip(cut_indices[:-1], cut_indices[1:]):
        single_line_images.append(image[i1: i2])

    return single_line_images


def bfs(img: List, x: int, y: int, used: List, h: int, w: int):
    q = set()
    q.add((x, y))

    while len(q) != 0:
        i, j = q.pop()

        used[i][j] = True

        if i > 0 and img[i - 1][j] and not used[i - 1][j]:
            q.add((i - 1, j))

        if i < h - 1 and img[i + 1][j] and not used[i + 1][j]:
            q.add((i + 1, j))

        if j > 0 and img[i][j - 1] and not used[i][j - 1]:
            q.add((i, j - 1))

        if j < w - 1 and img[i][j + 1] and not used[i][j + 1]:
            q.add((i, j + 1))


def cut_image_into_text_lines(image: np.array):
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

    rot_image = rotate_and_cut_off(image, s, (image.shape[1] // 2, image.shape[0] // 2))

    single_line_images = cut_horizontal_image(rot_image)

    return single_line_images
