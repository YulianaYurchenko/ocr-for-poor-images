from exceptions import UnableToLoadModel
from base import OCRModel
import numpy as np
import cv2 as cv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model

from preprocessing_tools import img_width, img_height
from preprocessing_tools import encode_single_image, decode_batch_predictions, cut_image_into_text_lines


class StandartModel(OCRModel):

    keras_model = None

    def model_prediction_text(self, image: np.array):
        preds = self.keras_model.predict(image)
        pred_lines = decode_batch_predictions(preds, max_label_len=32)

        pred_text = ''
        for i in range(len(pred_lines)):
            line = pred_lines[i]
            # Remove unidentified characters
            while line.find('[UNK]') != -1:
                line = line.replace('[UNK]', '')
            # Remove extra spaces from the end
            while line[-1] == ' ':
                line = line[:-1]
            if i != len(pred_lines) - 1:
                pred_text += line + '\n'
            else:
                pred_text += line + '\f'

        return pred_text

    def load(self):
        if not self._loaded:
            try:
                model = load_model('../models/pro_model.h5')
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
        # 2. Cut to single line images
        single_line_images = cut_image_into_text_lines(image)
        # 3. Resize to suit the model
        single_line_images = [cv.resize(img, (img_width, img_height)) for img in single_line_images]

        # 4. Some more preprocessing
        # (reshaping, normalizing, transposing (time dimension must correspond to the width of the image))
        single_line_images = [encode_single_image(img) for img in single_line_images]

        batch_size = len(single_line_images)
        single_line_images = tf.reshape(single_line_images, [batch_size, img_width, img_height, 1])
        return single_line_images
