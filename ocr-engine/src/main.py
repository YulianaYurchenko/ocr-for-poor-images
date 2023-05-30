import numpy as np
from PIL import Image
import cv2 as cv
import json

from tesseract import TESSERACT
from single_line_model import SingleLineModel
from standart_model import StandartModel
from metrics import edit_distance


with open(f'../../SingleLineImages/Test/ClearImages/clear_images_data.json') as json_file:
    single_line_DATA = json.load(json_file)

with open(f'../../StandartImages/Test/CrearImages/clear_images_data.json') as json_file:
    standart_DATA = json.load(json_file)


def take_clear_image_text(DATA, clear_image_name):
    res = ''
    clear_image_data = DATA[clear_image_name]

    for word_data in clear_image_data:
        word = str(word_data['word'])
        if word.find('\n') == -1:
            res += word + ' '
        else:
            res += word
    res = res[:-1]
    res += '\f'
    return res


def save_image(image: np.array, filename):
    img = Image.fromarray(np.uint8(image))
    img.save(filename)


def show_image(image: np.array):
    cv.imshow('img', image)
    cv.waitKey(0)


model1 = TESSERACT(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
model1.load()

model2 = SingleLineModel()
model2.load()

model3 = StandartModel()
model3.load()


single_line_image1 = cv.imread('../sample_images/single_line_degraded_image0.png')
single_line_image2 = cv.imread('../sample_images/single_line_degraded_image1.png')
single_line_images = [single_line_image1, single_line_image2]


for text in model1.recognize_text(single_line_images):
    print(f'Prediction: {text}')


for text in model2.recognize_text(single_line_images):
    print(f'Prediction: {text}')


test_images = []
for i in range(1000):
    path = f'../../SingleLineImages/Test/DegradedImages/degraded_image{i}.png'
    test_images.append(cv.imread(path))

mean_relative_mistake = 0
i = 0
for pred_text in model2.recognize_text(test_images):
    true_text = take_clear_image_text(single_line_DATA, f'clear_image{i}.png')

    relative_mistake = int(edit_distance(pred_text, true_text)) / len(true_text)
    relative_mistake = min(1., relative_mistake)
    
    if i % 100 == 0:
        print(f'True: {true_text}')
        print(f'Prediction: {pred_text}')
        print(f'Relative mistake: {relative_mistake}')
    
    mean_relative_mistake += relative_mistake
    i += 1

mean_relative_mistake /= len(test_images)
print(f'SingleLineModel mean relative mistake: {mean_relative_mistake}')


standart_image1 = cv.imread('../sample_images/degraded_image0.png')
standart_image2 = cv.imread('../sample_images/degraded_image1.png')
standart_images = [standart_image1, standart_image2]

for text in model1.recognize_text(standart_images):
    print(f'Prediction: {text}')


for text in model3.recognize_text(standart_images):
    print(f'Prediction: {text}')


test_images = []
for i in range(1000):
    path = f'../../StandartImages/Test/DegradedImages/degraded_image{i}.png'
    test_images.append(cv.imread(path))

mean_relative_mistake = 0
i = 0
for pred_text in model3.recognize_text(test_images):
    true_text = take_clear_image_text(standart_DATA, f'clear_image{i}.png')

    relative_mistake = int(edit_distance(pred_text, true_text)) / len(true_text)
    relative_mistake = min(1., relative_mistake)

    if i % 100 == 0:
        print(f'True: {true_text}')
        print(f'Prediction: {pred_text}')
        print(f'Relative mistake: {relative_mistake}')

    mean_relative_mistake += relative_mistake
    i += 1

mean_relative_mistake /= len(test_images)
print(f'StandartModel mean relative mistake: {mean_relative_mistake}')

