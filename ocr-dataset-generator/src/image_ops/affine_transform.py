from image_ops.base import BaseImageOperation
from transformations import affine_transform


class AffineTransformOperation(BaseImageOperation):
    """
    Class that implements affine transformation operation of an image
    """
    def __init__(self, pts1, pts2):
        self._op = lambda X: affine_transform(X, pts1, pts2)
