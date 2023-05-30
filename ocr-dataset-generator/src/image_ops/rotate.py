from image_ops.base import BaseImageOperation
from transformations import rotate


class RotateOperation(BaseImageOperation):
    """
    Class that implements operation of rotating an image
    """
    def __init__(self, angle, center: (int, int)):
        self._op = lambda X: rotate(X, angle, center)
