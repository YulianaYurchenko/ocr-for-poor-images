from exceptions import UnableToLoadModel
from base import OCRModel
import numpy as np
import cv2 as cv

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from preprocessing_tools import IMG_WIDTH, IMG_HEIGHT
from preprocessing_tools import encode_single_image, decode_batch_predictions, cut_image_into_text_lines


class CTCLayer(layers.Layer):
    def __init__(self, name=None, **kwargs):
        super(CTCLayer, self).__init__(name=name)
        super(CTCLayer, self).__init__(**kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def get_config(self):
        config = super(CTCLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


class StandartModel(OCRModel):

    _keras_model = None

    def _predict_multi_line_text(self, single_line_images: np.array):
        preds = self._keras_model.predict(single_line_images, verbose=0)
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
                model = load_model('../../models/pro_model.h5', custom_objects={'CTCLayer': CTCLayer})
                self._keras_model = keras.models.Model(
                    model.get_layer(name="image").input, model.get_layer(name="dense_1").output
                )
                self._model = self._predict_multi_line_text
            except Exception as ex:
                raise UnableToLoadModel(ex)
            else:
                self._loaded = True

    @staticmethod
    def preprocess_image(image: np.array) -> np.array:
        # 1. Cut to single line images
        single_line_images = cut_image_into_text_lines(image)
        # 2. Resize to suit the model
        single_line_images = [cv.resize(img, (IMG_WIDTH, IMG_HEIGHT)) for img in single_line_images]

        # 3. Some more preprocessing
        # (reshaping, normalizing, transposing (time dimension must correspond to the width of the image))
        single_line_images = [encode_single_image(img) for img in single_line_images]

        batch_size = len(single_line_images)
        single_line_images = tf.reshape(single_line_images, [batch_size, IMG_WIDTH, IMG_HEIGHT, 1])
        return single_line_images
