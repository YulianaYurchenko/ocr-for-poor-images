import numpy as np
import cv2 as cv


def resize(image: np.array, width: int, height: int, interpolation=cv.INTER_LINEAR) -> np.array:
    """
    Function resizes an image

    Args:
        image (np.array): tensor representing an image.
        width (int): the width of image you want to get.
        height (int): the height of image you want to get.
        interpolation: use cv.INTER_AREA for shrinking and cv.INTER_CUBIC (slow) & cv.INTER_LINEAR for zooming.
            By default, the interpolation method cv.INTER_LINEAR is used for all resizing purposes.
    """
    resized = cv.resize(image, (width, height), interpolation)
    return resized


def translate(image: np.array, twidth: int, theight: int) -> np.array:
    """
    Function translates (shifts) an image by (twidth, theight) in (x, y) direction
    """
    rows, cols = image.shape[:2]
    M = np.float32([[1, 0, twidth], [0, 1, theight]])
    translated = cv.warpAffine(image, M, (cols, rows))
    return translated


def rotate(image: np.array, angle: float, center: (int, int)) -> np.array:
    """
    Function rotates an image

    Args:
        image (np.array): tensor representing an image.
        angle (float): the angle of rotation is measured in degrees.
        center ((int, int)): center of rotation.
    """
    height, width = image.shape[:2]
    x, y = center

    cos_t = np.math.cos(angle)
    sin_t = np.math.sin(angle)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t + width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    rotated = cv.warpAffine(image, M, (new_width, new_height), borderValue=(255, 255, 255))
    return rotated


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

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(-height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t - width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    rotated = cv.warpAffine(image, M, (new_width, new_height))
    return rotated


def scale(image: np.array, kwidth: float, kheight: float, center: (int, int)) -> np.array:
    """
    Function scales an image

    Args:
        image (np.array): tensor representing an image.
        kwidth (float): scale kwidth times in x direction.
        kheight (float): scale kheight times in y direction.
        center ((int, int)): center of scaling.
    """
    rows, cols = image.shape[:2]
    x, y = center
    M = np.float32([[kwidth, 0, x * (1 - kwidth)], [0, kheight, y * (1 - kheight)]])
    scaled = cv.warpAffine(image, M, (cols, rows))
    return scaled


def affine_transform(image: np.array, pts1: np.float32([[], [], []]), pts2: np.float32([[], [], []])) -> np.array:
    """
    Function makes affine transformation of an image

    Args:
        image (np.array): tensor representing an image.
        pts1: three from the input image.
        pts2: their corresponding locations in the output image.
    """
    rows, cols = image.shape[:2]
    M = cv.getAffineTransform(pts1, pts2)
    transformed = cv.warpAffine(image, M, (cols, rows))
    return transformed
