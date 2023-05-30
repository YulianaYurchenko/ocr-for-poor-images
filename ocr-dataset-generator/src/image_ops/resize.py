from image_ops.base import BaseImageOperation
from transformations import resize
import cv2 as cv


class ResizeOperation(BaseImageOperation):
    """
    Class that implements operation of resizing an image
    """
    def __init__(self, width, height, interpolation=cv.INTER_LINEAR):
        self._op = lambda X: resize(X, width, height, interpolation)
