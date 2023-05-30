import os
import json
from numpy.random import randint

from PIL import Image, ImageFont, ImageDraw


def generate_single_line_clear_images(text_file_dir: str,
                                      output_dir: str,
                                      WIDTH: int,
                                      HEIGHT: int,
                                      N: int,
                                      font_size_min: int, font_size_max: int) -> None:

    if not (os.path.exists(text_file_dir) and os.path.isdir(output_dir)):
        print(text_file_dir, output_dir)
        raise ValueError('Invalid text file path and/or output directory path.')

    toJson = {}

    text = open(text_file_dir, 'r').read().split()
    word_index = 0

    for i in range(N):
        font_size = randint(font_size_min, font_size_max)
        font = ImageFont.truetype(font="Windows/Fonts/Arial/arial.ttf",
                                  size=font_size, encoding='unicode')

        img = Image.new(mode='RGB', size=(WIDTH, HEIGHT), color=(256, 256, 256))
        draw = ImageDraw.Draw(img)

        x = randint(0, font_size)
        y = randint(0, HEIGHT - font_size)
        words_data = []
        able_to_place_text = True
        while able_to_place_text:
            if word_index == 0:
                word = 'Title'
            else:
                word = text[word_index]
            word_width, word_height = font.getsize(word)
            space_width, space_height = font.getsize(' ')

            if x + word_width + space_width > WIDTH:
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

        toJson.update({f'clear_image{i}.png': words_data})

    with open(os.path.join(output_dir, 'clear_images_data.json'), 'w') as write_file:
        json.dump(toJson, write_file, indent=4)


if __name__ == '__main__':
    generate_single_line_clear_images(text_file_dir="../data/gothicliterature_filtered.txt",
                                      output_dir="../../ClearImages/data",
                                      WIDTH=256,
                                      HEIGHT=32,
                                      N=11000,
                                      font_size_min=22,
                                      font_size_max=26)
