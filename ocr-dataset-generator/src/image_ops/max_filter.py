from image_ops.base import BaseImageOperation
from blur_images import max_filter


class MaxFilterOperation(BaseImageOperation):
    """
    Class that implements operation of adding max filter to an image.
    """
    def __init__(self, radius=3):
        self._op = lambda X: max_filter(X, radius)

