import re
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

from pathlib import Path
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

import json
import cv2 as cv

#                   Load the data

with open('../../StandartImages/Train/ClearImages/clear_images_data.json') as json_file:
    DATA = json.load(json_file)


def take_clear_image_text(clear_image_name):
    res = ''
    clear_image_data = DATA[clear_image_name]

    for word_data in clear_image_data:
        word = str(word_data['word'])
        if word.find('\n') == -1:
            res += word + ' '
        else:
            res += word
    return res[:-1]


# Path to the data directory
data_dir = Path('../../StandartImages/Train/DegradedImages/')

# Get list of all the images
image_paths = list(map(str, list(data_dir.glob("*.png"))))
image_paths = sorted(image_paths, key=lambda x: int(''.join(re.findall('\d+', x))))
labels = [take_clear_image_text('clear_image' + str(i) + '.png') for i in range(len(image_paths))]

characters = ['\n', ' ', '!', '"', "'", '(', ')', '*', ',', '-', '.', '/',
              '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '=', '?',
              'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
              'V', 'W', 'X', 'Y', 'Z',
              'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
              'v', 'w', 'x', 'y', 'z']

print("Number of images found: ", len(image_paths))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)


#                   Preprocessing

def rotate_and_cut_off(image: np.array, angle: float, center: (int, int)) -> np.array:
    height, width = image.shape[:2]
    x, y = center

    theta = angle / 180.0 * np.math.pi
    cos_t = np.math.cos(theta)
    sin_t = np.math.sin(theta)
    M = np.float32([[cos_t, sin_t, x - x * cos_t - y * sin_t], [-sin_t, cos_t, y + x * sin_t - y * cos_t]])

    new_width = int(-height * np.abs(sin_t) + width * cos_t)
    new_height = int(height * cos_t - width * np.abs(sin_t))

    M[0, 2] += (new_width / 2) - x
    M[1, 2] += (new_height / 2) - y

    rotated = cv.warpAffine(image, M, (new_width, new_height))
    return rotated


def make_image_horizontal(image: np.array, max_angle=5):
    height, width = image.shape[:2]

    max_variation = 0
    best_angle = None

    for angle in np.linspace(-max_angle, max_angle, 21):
        M = cv.getRotationMatrix2D((width // 2, height // 2), angle, 1)
        rotated_img = cv.warpAffine(image, M, (width, height))

        x = [sum(1 - row / 255) for row in rotated_img]
        x_mean = sum(x) / len(x)
        x_RMSE = sqrt(sum((x - x_mean)**2) / len(x))
        x_variation = x_RMSE / x_mean

        if x_variation > max_variation:
            best_angle = angle
            max_variation = x_variation

    horizontal_img = rotate_and_cut_off(image, best_angle, (width // 2, height // 2))
    return horizontal_img[2:-2]


def cut_image_into_text_lines(image: np.array, valley_coef=0.04, slope_coef=0.02, deviation_bound=0.3):
    """
        Function cuts an multi-line image to list of single-line images

        Args:
            image (np.array): tensor representing an image.
            valley_coef (float): relative height of areas of the image where few black pixels are
            slope_coef (float): relative height of areas of the image where number of black pixels starts to increase
            deviation_bound (float): hyperparameter that regulates the increase in number of black pixels where text appears
    """
    image = make_image_horizontal(image)

    height, width = image.shape[:2]
    valley_size = int(height * valley_coef)
    slope_size = int(height * slope_coef)

    x = [sum(1 - row / 255) for row in image]

    x_mean = sum(x) / len(x)
    x_RMSE = sqrt(sum((x - x_mean) ** 2) / len(x))

    prev_cut_index = -1
    cut_indices = [0]
    for i in range(valley_size + slope_size, len(x) - valley_size):
        maybe_valley = x[i - valley_size: i]
        mean = sum(maybe_valley) / len(maybe_valley)
        derivation = (sum(x[i: i + slope_size]) - sum(x[i - slope_size: i])) / slope_size

        if mean < x_mean - x_RMSE / 2 and derivation / x_RMSE > deviation_bound:
            if prev_cut_index == -1 or i - prev_cut_index > valley_size:
                cut_indices.append(i)
                prev_cut_index = i
    cut_indices.append(height)

    single_line_images = []
    for j in range(len(cut_indices) - 1):
        single_line_images.append(image[cut_indices[j]: cut_indices[j + 1]])

    return single_line_images


# Batch size for training and validation
batch_size = 16

# Desired image dimensions
img_width = 256
img_height = 32


single_line_images = []
single_line_labels = []

errors = 0

for i in range(len(image_paths)):
    if i % 1000 == 0:
        print('#')
    image = cv.imread(image_paths[i])
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    sngl_ln_imgs = cut_image_into_text_lines(image)
    sngl_ln_lbls = labels[i].split('\n')
    if len(sngl_ln_imgs) != len(sngl_ln_lbls):
        errors += 1
        continue
    else:
        single_line_images = [cv.resize(img, (img_width, img_height)) for img in sngl_ln_imgs]
        single_line_labels = sngl_ln_lbls
print(f'{errors} images are lost')
print(f'There are {len(single_line_images)} images now')


# Maximum length of any captcha in the dataset
max_label_len = max([len(label) for label in single_line_labels])

for i in range(len(single_line_labels)):
    single_line_labels[i] += ' ' * (max_label_len - len(single_line_labels[i]))


# Mapping characters to integers
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


def split_data(images, labels, train_size=0.9, shuffle=True):
    # 1. Get the total size of the dataset
    size = len(images)
    # 2. Make an indices array and shuffle it, if required
    indices = np.arange(size)
    if shuffle:
        np.random.shuffle(indices)
    # 3. Get the size of training samples
    train_samples = int(size * train_size)
    # 4. Split data into training and validation sets
    x_train, y_train = images[indices[:train_samples]], labels[indices[:train_samples]]
    x_valid, y_valid = images[indices[train_samples:]], labels[indices[train_samples:]]
    return x_train, x_valid, y_train, y_valid


# Splitting data into training and validation sets
x_train, x_valid, y_train, y_valid = split_data(np.array(single_line_images), np.array(single_line_labels))


def encode_single_sample(image, label):
    # 1. Convert grayscale image to 3-dimensional tensor
    image = tf.reshape(image, [image.shape[0], image.shape[1], 1])
    # 2. Convert to float32 in [0, 1] range
    image = tf.image.convert_image_dtype(image, tf.float32)
    # 3. Transpose the image because we want the time
    # dimension to correspond to the width of the image.
    image = tf.transpose(image, perm=[1, 0, 2])
    # 4. Map the characters in label to numbers
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    # 5. Return a dict as our model is expecting two inputs
    return {"image": image, "label": label}


#                   Create Database objects

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)


#                   Visualize the data

_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    imgs = batch["image"]
    lbls = batch["label"]
    for i in range(16):
        img = (imgs[i] * 255).numpy().astype("uint8")
        lbl = tf.strings.reduce_join(num_to_char(lbls[i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(lbl, size=6)
        ax[i // 4, i % 4].axis("off")
plt.show()


#                   Model

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


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=max_label_len, dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used three max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller.
    # The number of filters in the last layer is 64.
    # Reshape accordingly before passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(len(characters) + 1, activation="softmax", name="dense2")(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt)
    return model


# Get the model
model = build_model()
model.summary()


#                   Training

epochs = 100
early_stopping_patience = 20
# Add early stopping
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=epochs,
    callbacks=[early_stopping],
    validation_data=validation_dataset,
    verbose=1
)


#                   Saving the model

if os.path.isfile('./standart_model') is False:
    model.save('./standart_model')


'''
#                   Loading the model

model = load_model('./single_line_model')
'''

#                   Inference

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_label_len]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


#  Let's check results on some validation samples
for batch in validation_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(16):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        label = tf.strings.reduce_join(num_to_char(batch_labels[i])).numpy().decode("utf-8")
        title = f"True:\n {label}\n\nPrediction:\n {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title, size=6)
        ax[i // 4, i % 4].axis("off")
    break
plt.show()
