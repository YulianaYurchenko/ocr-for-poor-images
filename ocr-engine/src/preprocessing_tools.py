import numpy as np
import cv2 as cv
from math import sqrt
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Desired image dimensions
img_width = 256
img_height = 32

# All characters that can happen in labels
characters = ['\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']


def rotate_and_cut_off(image: np.array, angle: float, center: (int, int)) -> np.array:
    """
        Function rotates an image and cuts off black borders

        Args:
            image (np.array): tensor representing an image.
            angle (float): the angle of rotation is measured in degrees.
            center ((int, int)): center of rotation.
    """
    height, width = image.shape[:2]
    x, y = center

    theta = angle / 180.0 * np.math.pi
    cos_t = np.math.cos(theta)
    sin_t = np.math.sin(theta)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(-height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t - width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    rotated = cv.warpAffine(image, M, (new_width, new_height))
    return rotated


def make_image_horizontal(image: np.array, max_angle=5) -> np.array:
    """
        Function rotates an image to make its text horizontal

        Args:
            image (np.array): tensor representing an image.
            max_angle (float): the max rotating angle to find the horizontal position
    """
    height, width = image.shape[:2]

    max_variation = 0
    best_angle = None

    for angle in np.linspace(-max_angle, max_angle, 21):
        M = cv.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        rotated_img = cv.warpAffine(image, M, (width, height))

        x = [sum(1 - row / 255) for row in rotated_img]
        x_mean = sum(x) / len(x)
        x_RMSE = sqrt(sum((x - x_mean)**2) / len(x))
        x_variation = x_RMSE / x_mean

        if x_variation > max_variation:
            best_angle = angle
            max_variation = x_variation

    horizontal_img = rotate_and_cut_off(image, best_angle, (width // 2, height // 2))
    return horizontal_img[2:-2]


def cut_image_into_text_lines(image: np.array, valley_coef=0.04, slope_coef=0.02, deviation_bound=0.3, show_plot=False):
    """
        Function cuts an multi-line image to list of single-line images

        Args:
            image (np.array): tensor representing an image.
            valley_coef (float): relative height of areas of the image where few black pixels are
            slope_coef (float): relative height of areas of the image where number of black pixels starts to increase
            deviation_bound (float): hyperparameter that regulates the increase in number of black pixels where text appears
    """
    image = make_image_horizontal(image)

    height, width = image.shape[:2]
    valley_size = int(height * valley_coef)
    slope_size = int(height * slope_coef)

    x = [sum(1 - row / 255) for row in image]

    x_mean = sum(x) / len(x)
    x_RMSE = sqrt(sum((x - x_mean) ** 2) / len(x))

    prev_cut_index = -1
    cut_indices = [0]
    for i in range(valley_size + slope_size, len(x) - valley_size):
        maybe_valley = x[i - valley_size: i]
        mean = sum(maybe_valley) / len(maybe_valley)
        derivation = (sum(x[i: i + slope_size]) - sum(x[i - slope_size: i])) / slope_size

        if mean < x_mean - x_RMSE / 2 and derivation / x_RMSE > deviation_bound:
            if prev_cut_index == -1 or i - prev_cut_index > valley_size:
                cut_indices.append(i)
                prev_cut_index = i
                if show_plot:
                    plt.plot([i, i], [0, 100], 'k--')
            else:
                if show_plot:
                    plt.plot([i, i], [0, 100], 'r-')
    cut_indices.append(height)

    if show_plot:
        plt.plot([i for i in range(len(x))], x)
        plt.show()

    single_line_images = []
    for j in range(len(cut_indices) - 1):
        single_line_images.append(image[cut_indices[j]: cut_indices[j + 1]])

    return single_line_images


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

    for word_data in clear_image_data:
        word = str(word_data['word'])
        if word.find('\n') == -1:
            res += word + ' '
        else:
            res += word
    res = res[:-1]
    res += '\f'
    return res
