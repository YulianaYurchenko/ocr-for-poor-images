from image_ops.base import BaseImageOperation
from blur_images import min_filter


class MinFilterOperation(BaseImageOperation):
    """
    Class that implements operation of adding min filter to an image.
    """
    def __init__(self, radius=3):
        self._op = lambda X: min_filter(X, radius)

