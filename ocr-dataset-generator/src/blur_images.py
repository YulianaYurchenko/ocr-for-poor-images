import PIL
from PIL import Image
from PIL import ImageFilter
import numpy as np


def blur_images(filename: str, filter: str, radius: int):

    my_image = Image.open(filename)

    if filter == "gaussian":
        result_image = my_image.filter(ImageFilter.GaussianBlur(radius))

    elif filter == "box":
        result_image = my_image.filter(ImageFilter.BoxBlur(radius))

    elif filter == "min":
        result_image = my_image.filter(ImageFilter.MinFilter(radius))

    elif filter == "max":
        result_image = my_image.filter(ImageFilter.MaxFilter(radius))

    elif filter == "median":
        result_image = my_image.filter(ImageFilter.MedianFilter(radius))

    else:
        raise NotImplementedError()

    return result_image


def gaussian_blur(image: np.array, radius=1) -> np.array:
    image = np.clip(image, 0, 255)
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(blured)


def box_blur(image: np.array, radius=1) -> np.array:
    image = np.clip(image, 0, 255)
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.BoxBlur(radius))
    return np.asarray(blured)


def min_filter(image: np.array, radius=3) -> np.array:
    image = np.clip(image, 0, 255)
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.MinFilter(radius))
    return np.asarray(blured)


def max_filter(image: np.array, radius=3) -> np.array:
    image = np.clip(image, 0, 255)
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.MaxFilter(radius))
    return np.asarray(blured)


def median_filter(image: np.array, radius=3) -> np.array:
    image = np.clip(image, 0, 255)
    blured = PIL.Image.fromarray(np.uint8(image)).filter(ImageFilter.MedianFilter(radius))
    return np.asarray(blured)
