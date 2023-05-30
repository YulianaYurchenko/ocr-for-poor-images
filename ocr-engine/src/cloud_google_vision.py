import json
from metrics import edit_distance


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
    return res


mean_relative_mistake = 0
toJson = {}
for i in range(33, 50):
    print(f'degraded_image{i}.png')
    true_text = take_clear_image_text(standart_DATA, f'clear_image{i}.png')
    google_vision_text = input('Enter Cloud Google Vision prediction\n')
    while google_vision_text.find(' ') != -1:
        google_vision_text = google_vision_text.replace(' ', '')
    print(google_vision_text)
    google_vision_text = input('\n')
    relative_mistake = int(edit_distance(google_vision_text, true_text)) / len(true_text)
    mean_relative_mistake += relative_mistake

    toJson.update({f'degraded_image{i}.png': {'true text': true_text,
                                              'Cloud Google Vision text': google_vision_text,
                                              'relative Levenshtein distance': relative_mistake}})

mean_relative_mistake /= 90.
toJson.update({'mean relative Levenshtein distance': mean_relative_mistake})

with open('./cloud_google_vision_results.json', 'w') as write_file:
    json.dump(toJson, write_file, indent=4)



