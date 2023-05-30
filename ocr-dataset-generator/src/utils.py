import numpy as np


def scale_point2d(src_point: (int, int), original_size: (int, int), target_size: (int, int)):
    orig_width, orig_height = original_size
    target_width, target_height = target_size

    orig_x, orig_y = src_point
    scaled_x = orig_x / orig_width * target_width
    scaled_y = orig_y / orig_height * target_height

    return round(scaled_x), round(scaled_y)


def rotate_point2d(src_point: (int, int), angle: float, center: (int, int), radians=False):

    if not radians:
        angle *= np.pi / 180

    (x1, y1), (x0, y0) = src_point, center
    rotated_x = (x1 - x0) * np.cos(angle) + (y1 - y0) * np.sin(angle) + x0
    rotated_y = - (x1 - x0) * np.sin(angle) + ((y1 - y0) * np.cos(angle)) + y0

    return round(rotated_x), round(rotated_y)


def rotate_point2d_in_not_cut_img(src_point: (int, int), angle: float, center: (int, int), img_size: (int, int)) -> (int, int):
    width, height = img_size
    x0, y0 = center
    x, y = src_point

    theta = angle / 180.0 * np.math.pi
    cos_t = np.math.cos(theta)
    sin_t = np.math.sin(theta)

    rotated_x = (x - x0) * cos_t + (y - y0) * sin_t + x0
    rotated_y = - (x - x0) * sin_t + (y - y0) * cos_t + y0

    new_width = int(height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t + width * np.abs(sin_t))

    rotated_x += (new_width / 2) - x0
    rotated_y += (new_height / 2) - y0

    return round(rotated_x), round(rotated_y)
