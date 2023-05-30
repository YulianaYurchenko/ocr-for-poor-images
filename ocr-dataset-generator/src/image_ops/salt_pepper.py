from image_ops.base import BaseImageOperation
from noise import NoiseTypes, noisify


class SaltPepperOperation(BaseImageOperation):
    """
    Class that implements operation of adding salt and pepper noise to an image.
    """
    def __init__(self, salt_vs_pepper=0.5, amount=0.01):
        self._op = lambda X: noisify(X, noise_type=NoiseTypes.SALT_AND_PEPPER,                  
                                     salt_vs_pepper=salt_vs_pepper, amount=amount)
