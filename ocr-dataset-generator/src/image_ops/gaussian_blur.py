from image_ops.base import BaseImageOperation
from blur_images import gaussian_blur


class GaussianBlurOperation(BaseImageOperation):
    """
    Class that implements operation of adding gaussian blur to an image.
    """
    def __init__(self, radius=1):
        self._op = lambda X: gaussian_blur(X, radius)

