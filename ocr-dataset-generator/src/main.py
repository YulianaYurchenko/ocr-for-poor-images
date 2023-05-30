import json
from tqdm import tqdm

from PIL import Image
from numpy.random import randint
from numpy.random import uniform
from numpy.random import choice
import numpy as np
import cv2 as cv
from functools import reduce
from image_ops import *
import pytesseract as tesseract
from metrics import edit_distance

from utils import scale_point2d, rotate_point2d_in_not_cut_img

np.random.seed(0)

input_dir = '../../ClearImages/data'
output_dir = '../../DegradedImages/data'

with open(f'{input_dir}/clear_images_data.json') as json_file:
    DATA = json.load(json_file)


def take_clear_image_text(clear_image_name):
    res = ''
    clear_image_data = DATA[clear_image_name]

    for word_data in clear_image_data['words']:
        word = str(word_data['word'])
        if word.find('\n') == -1:
            res += word + ' '
        else:
            res += word
    res += '\f'
    return res


def take_degraded_image_bboxes(clear_image_name, orig_size, target_size, angle):
    words = []
    clear_image_data = DATA[clear_image_name]

    for word_data in clear_image_data['words']:
        word = word_data['word']
        orig_bbox = word_data['coord']
        new_bbox = []
        for point in orig_bbox:
            scaled_point = scale_point2d(src_point=point, original_size=orig_size, target_size=target_size)
            new_point = rotate_point2d_in_not_cut_img(src_point=scaled_point, angle=angle,
                                                      center=(target_size[0] // 2, target_size[1] // 2),
                                                      img_size=target_size)
            new_bbox.append(new_point)
        words.append({'word': word, 'coord': new_bbox})
    return words


tesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

Interpolations = [cv.INTER_AREA, cv.INTER_LINEAR, cv.INTER_CUBIC]

toJson = {}
tesseract_mean_relative_mistake = 0

N = 11000

for i in tqdm(range(N)):

    clear_image_name = f'clear_image{i}.png'
    clear_image = cv.imread(f'{input_dir}/' + clear_image_name, cv.IMREAD_GRAYSCALE)
    new_data = {}

    orig_height, orig_width = clear_image.shape[:2]
    orig_size = orig_width, orig_height
    target_size = target_width, target_height = randint(orig_width // 1.5, orig_width * 1.5), randint(
        orig_height // 1.5, orig_height * 1.5)

    angle = uniform(-10, 10)

    gaussian_noise_std = uniform(0., 32.)
    speckle_std = uniform(0., 0.6)
    salt_amount = uniform(0, 0.1)
    gaussian_blur_radius = randint(0, 2)
    box_blur_radius = randint(0, 2)
    min_filter_radius = choice([1, 3], p=[0.75, 0.25])
    max_filter_radius = 1 if min_filter_radius == 1 and salt_amount > 0.01 else choice([1, 3])
    salt_vs_pepper = uniform(0., 1.)
    salt_pepper_amount = 0. if max_filter_radius == 3 else choice([0., 0.01], p=[0.75, 0.25])

    degr_pipeline = \
        [ResizeOperation(width=target_width, height=target_height, interpolation=Interpolations[randint(0, 3)]),
         RotateOperation(angle=np.radians(angle), center=(target_width // 2, target_height // 2)),
         SpeckleOperation(mean=0, stddev=speckle_std),
         GaussianNoiseOperation(mean=0, stddev=gaussian_noise_std),
         SaltPepperOperation(salt_vs_pepper=1., amount=salt_amount),
         GaussianBlurOperation(radius=gaussian_blur_radius),
         BoxBlurOperation(radius=box_blur_radius),
         MinFilterOperation(radius=min_filter_radius),
         MaxFilterOperation(radius=max_filter_radius),
         SaltPepperOperation(salt_vs_pepper=salt_vs_pepper, amount=salt_pepper_amount),
         ]

    degraded = reduce(lambda image, op: op(image), degr_pipeline, clear_image)
    degraded = np.clip(degraded, 0, 255)
    degraded = np.uint8(degraded)

    operation = ResizeOperation(width=orig_width, height=orig_height, interpolation=cv.INTER_CUBIC)
    resized_back_without_rotation = operation(degraded)

    psnr = cv.PSNR(clear_image, resized_back_without_rotation)

    Image.fromarray(degraded).save(f'{output_dir}/degraded_image{i}.png')

    real_text = take_clear_image_text(clear_image_name)
    tesseract_text = tesseract.image_to_string(degraded)
    tesseract_absolute_mistake = int(edit_distance(real_text, tesseract_text))
    tesseract_relative_mistake = tesseract_absolute_mistake / len(real_text)
    tesseract_mean_relative_mistake += tesseract_relative_mistake

    words = take_degraded_image_bboxes(clear_image_name, orig_size, target_size, angle)

    new_data.update({'target_width': int(target_width),
                     'target_height': int(target_height),
                     'angle': float(angle),
                     'gaussian_noise_std': float(gaussian_noise_std),
                     'speckle_std': float(speckle_std),
                     'salt_amount': float(salt_amount),
                     'salt_vs_pepper': float(salt_vs_pepper),
                     'salt_pepper_amount': float(salt_pepper_amount),
                     'gaussian_blur_radius': int(gaussian_blur_radius),
                     'box_blur_radius': int(box_blur_radius),
                     'min_filter_radius': int(min_filter_radius),
                     'max_filter_radius': int(max_filter_radius),
                     'PSNR': psnr,
                     'tesseract_output': tesseract_text.split('\n'),
                     'tesseract_absolute_mistake': tesseract_absolute_mistake,
                     "tesseract_relative_mistake": tesseract_relative_mistake,
                     'n': DATA[clear_image_name]['n'],
                     'words': words})
    toJson.update({f'degraded_image{i}.png': new_data})


tesseract_mean_relative_mistake /= N
print(tesseract_mean_relative_mistake)
toJson.update({'tesseract_mean_relative_mistake': tesseract_mean_relative_mistake})

with open(f'{output_dir}/degraded_images_data.json', 'w') as write_file:
    json.dump(toJson, write_file, indent=4)
