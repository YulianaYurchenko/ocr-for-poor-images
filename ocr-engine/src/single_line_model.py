from exceptions import UnableToLoadModel
from base import OCRModel
import numpy as np
import cv2 as cv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from preprocessing_tools import encode_single_image, decode_batch_predictions, img_width, img_height


class SingleLineModel(OCRModel):

    keras_model = None

    def model_prediction_text(self, image: np.array):
        preds = self.keras_model.predict(image)
        pred_text = decode_batch_predictions(preds, max_label_len=30)[0]
        # Remove unidentified characters
        while pred_text.find('[UNK]') != -1:
            pred_text = pred_text.replace('[UNK]', '')
        # Remove extra spaces from the end
        while pred_text[-1] == ' ':
            pred_text = pred_text[:-1]
        pred_text += '\f'
        return pred_text

    def load(self):
        if not self._loaded:
            try:
                model = load_model('../models/single_line_model')
                self.keras_model = keras.models.Model(
                    model.get_layer(name="image").input, model.get_layer(name="dense2").output
                )
                self._model = self.model_prediction_text
            except Exception as ex:
                raise UnableToLoadModel(ex)
            else:
                self._loaded = True

    @staticmethod
    def preprocess_image(image: np.array) -> np.array:
        # 1. Convert to grayscale
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # 2. Resize to suit the model
        image = cv.resize(image, (img_width, img_height))
        # 3. Some more preprocessing
        image = encode_single_image(image)
        # 4. Batch size = 1
        image = tf.reshape(image, [1, img_width, img_height, 1])
        return image
