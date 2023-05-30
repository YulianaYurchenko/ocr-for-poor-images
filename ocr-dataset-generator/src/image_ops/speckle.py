from image_ops.base import BaseImageOperation
from noise import NoiseTypes, noisify


class SpeckleOperation(BaseImageOperation):
    """
    Class that implements operation of adding speckle noise to an image.
    """
    def __init__(self, mean=0.0, stddev=1.0):
        self._op = lambda X: noisify(X, noise_type=NoiseTypes.SPECKLE, mean=mean, stddev=stddev)

