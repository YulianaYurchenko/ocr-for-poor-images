from typing import List

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


def plot_x_bar(image: np.array):
    fig = plt.figure(figsize=(8, 6))

    x = np.sum(1 - image / 255, axis=1)

    plt.grid(zorder=0)
    plt.bar([i for i in range(image.shape[0])], x, zorder=13, linewidth=0.0, color='k')
    plt.xlabel('i -- Number of row', fontsize=20)
    plt.ylabel('x -- Amount of black', fontsize=20)
    return fig


def cut_image_into_text_lines(image: np.ndarray,
                              number_of_lines=7,
                              gradient_bound=0.3,
                              show_plot=False) -> List[np.ndarray]:
    """
    Function cuts a multi-line image to list of single-line images

    Args:
        image (np.ndarray): tensor representing an image
        number_of_lines (int): approximate number of text lines that image has
        gradient_bound (float): hyperparameter that regulates the increase
        in number of black pixels where text appears
    """

    if show_plot:
        plot_x_bar(image)

    c = 0
    q = 0

    height, width = image.shape[:2]
    # height of areas of the image where few black pixels are
    valley_size = int(height / number_of_lines / 3.5)
    # height of areas of the image where number of black pixels starts to increase
    slope_size = int(height / number_of_lines / 7.)

    x = np.sum(1 - image / 255, axis=1)
    prev_cut_index = -1
    cut_indices = [0]
    for i in range(valley_size + slope_size, len(x) - valley_size):
        valley = x[i - valley_size: i]
        gradient = (np.sum(x[i: i + slope_size]) - np.sum(x[i - slope_size: i])) / slope_size
        if np.mean(valley) < np.mean(x) - np.std(x) / 2 and gradient > gradient_bound * np.std(x):
            if prev_cut_index == -1 or i - prev_cut_index > valley_size:
                cut_indices.append(i)
                prev_cut_index = i
                if show_plot:
                    if c == 0:
                        plt.plot([i, i], [0, 135], 'k--', label='Final places of cuts')
                        c += 1
                    else:
                        plt.plot([i, i], [0, 135], 'k--')
            else:
                if show_plot:
                    if q == 0:
                        plt.plot([i, i], [0, 135], 'r-', alpha=0.5, label='Possible places of cuts')
                        q += 1
                    else:
                        plt.plot([i, i], [0, 135], 'r-', alpha=0.5)
    cut_indices.append(height)

    single_line_images = []
    for j in range(len(cut_indices) - 1):
        single_line_images.append(image[cut_indices[j]: cut_indices[j + 1]])

    plt.title('Detecting text lines', fontsize=24)
    plt.legend(fontsize=16)

    return single_line_images


def cut_image_into_text_lines_new(image: np.ndarray) -> List[np.ndarray]:

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

    return single_line_images, len(peaks)
