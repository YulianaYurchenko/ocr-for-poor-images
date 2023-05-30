import os
import json
from tqdm import tqdm

import numpy as np
from numpy.random import randint

from PIL import Image, ImageFont, ImageDraw


np.random.seed(0)


def generate_clear_images(text_file_dir: str,
                          output_dir: str,
                          line_spacing: float,
                          WIDTH: int,
                          HEIGHT: int,
                          N: int,
                          font_size_min: int,
                          font_size_max: int) -> None:

    if not (os.path.exists(text_file_dir) and os.path.isdir(output_dir)):
        print(text_file_dir, output_dir)
        raise ValueError('Invalid text file path and/or output directory path.')

    toJson = {}

    text = open(text_file_dir, 'r').read().split()
    word_index = 0

    for i in tqdm(range(N)):

        font_size = randint(font_size_min, font_size_max)
        font = ImageFont.truetype(font="Windows/Fonts/Arial/arial.ttf", size=font_size, encoding='unicode')

        img = Image.new(mode='RGB', size=(WIDTH, HEIGHT), color=(256, 256, 256))
        draw = ImageDraw.Draw(img)

        x = 10
        y = 10
        n = 0
        words_data = []
        able_to_place_text = True
        while able_to_place_text:
            word = text[word_index]
            word_width, word_height = font.getsize(word)
            space_width, _ = font.getsize(' ')

            if x + word_width + 10 > WIDTH:
                x = 10
                y += int(font_size * line_spacing)
                n += 1
                words_data[len(words_data) - 1]['word'] += '\n'

            if y + word_height + 10 > HEIGHT:
                able_to_place_text = False
                img.save(output_dir + "/clear_image" + str(i) + ".png")
            else:
                draw.text(xy=(x, y), text=word, fill=(0, 0, 0), font=font)
                words_data.append({
                    'word': word,
                    'coord': [(x, y),
                              (x + word_width, y),
                              (x + word_width, y + word_height),
                              (x, y + word_height)]
                })
                x += word_width + space_width
                word_index += 1

        toJson.update({f'clear_image{i}.png': {'n': n, 'words': words_data}})

    with open(os.path.join(output_dir, 'clear_images_data.json'), 'w') as write_file:
        json.dump(toJson, write_file, indent=4)


if __name__ == '__main__':
    generate_clear_images(text_file_dir="../data/gothicliterature_filtered.txt",
                          output_dir="../../ClearImages/data",
                          line_spacing=1.5,
                          WIDTH=256,
                          HEIGHT=256,
                          N=11000,
                          font_size_min=22,
                          font_size_max=26)
