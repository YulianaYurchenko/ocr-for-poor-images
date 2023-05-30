from abc import ABC, abstractproperty, abstractmethod, abstractstaticmethod, abstractstaticmethod
import numpy as np
from exceptions import ModelIsNotLoaded


class OCRModel(ABC):
    _loaded = False
    _model = None
    """
    Base OCR model that contains main interface methods.
    """

    def is_loaded(self) -> bool:
        """
        Returns whether the model is loaded or not.
        Returns:
            (bool): condition state
        """
        return self._loaded

    @abstractstaticmethod
    def preprocess_image(image: np.array) -> np.array:
        """
        Preprocesses an image to make it model ready.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Loads model's weights.
        """
        pass

    def recognize_text(self, images: [np.array]) -> [str]:
        """
        Runs images through the model to recognize text on them
        Args:
            images (np.array): sequence of images
        Returns:
            (str): sequence of images text
        """
        if not self.is_loaded():
            raise ModelIsNotLoaded

        for image in images:
            yield self._model(self.preprocess_image(image))

    def __call__(self, images: np.array) -> str:
        return self.recognize_text(images)
