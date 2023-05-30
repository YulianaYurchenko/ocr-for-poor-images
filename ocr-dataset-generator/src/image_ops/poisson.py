from .base import BaseImageOperation
from noise import NoiseTypes, noisify


class PoissonNoiseOperation(BaseImageOperation):
    """
    Class that implements operation of adding poisson noise to an image.
    """
    def __init__(self):
        self._op = lambda X: noisify(X, noise_type=NoiseTypes.POISSON)
