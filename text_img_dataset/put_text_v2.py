import PIL
from PIL import ImageFont, ImageDraw, Image
from os import listdir  # list of files
import random
import sys
import time
from xml_class import XML  # custom xml


BACKGROUNDS_PATH = 'backgrounds/'
FONTS_PATH = 'fonts/'
SAVE_PATH = 'data/Images/'
TRAIN_PATH = 'data/ImageSets/'
TEST_PATH = 'data/ImageSets/'
FAKE_WORDS_PATH = './fakewords.txt'
MAX_ITERATIONS = 10000

labels = ['aeroexpress', 'Microsoft', 'Apple', 'Google', ]


def load_fake_words(path=FAKE_WORDS_PATH):
    def filter_list(words):
        words = [word for word in words if (word not in labels) and (word.title() not in labels)]
        return words

    with open(FAKE_WORDS_PATH, 'r') as f:
        words = f.read()
        words += f.read().title()
        words = words.split()
    words = filter_list(words)
    return words


def expand_blocks(blocks, text_size):
    new_blocks = []
    for block in blocks:
        new_left_x = 1 if block[0] - text_size[0] <= 0 else block[0] - text_size[0]
        new_up_y = 1 if block[1] - text_size[1] <= 0 else block[1] - text_size[1]
        new_blocks.append([new_left_x, new_up_y, block[2], block[3]])

    return new_blocks


def create_x_y_from_blocks(blocks, image_size):
    all_possibles = []

    if len(blocks) == 0:
        return [(x, y) for x in range(1, image_size[0]) for y in range(1, image_size[1])]

    for block in blocks:
        all_possibles.append([(x, y) for x in range(1, image_size[0]) for y in range(1, image_size[1])
                              if x < block[0] or y < block[1] or x > block[2] or y > block[3]])

    if len(all_possibles) > 1:
        result = set(all_possibles[0]).intersection(set(all_possibles[1]))
        for i in range(2, len(all_possibles)):
            result = result.intersection(set(all_possibles[i]))
        return list(result)

    return all_possibles[0]


fakes = load_fake_words()

fonts = [FONTS_PATH + file for file in listdir(FONTS_PATH)]  # fonts path

backgrounds = [BACKGROUNDS_PATH + file for file in listdir(BACKGROUNDS_PATH)]  # files path

max_ratio = (2048, 2048)

random.seed(int(sys.argv[2]))

colours = [(r, g, b)
           for r in range(0, 256)
           for g in range(0, 256)
           for b in range(0, 256)]

all_images = []


def create_dataset(iterations, disp_step=1, last_saved=0):

    t = 0

    for i in range(last_saved, iterations + last_saved):
        start = time.time()

        img_name = '%05d' % i + '.jpg'
        background = random.choice(backgrounds)
        try:
            background = Image.open(background)
        except:
            print('Failed open image')
            continue
        background.thumbnail(max_ratio, PIL.Image.ANTIALIAS)
        width, height = background.size  # image size
        min_w_h = min(width, height)
        obj_quantity = random.randint(0, 4)
        fakes_quantity = random.randint(0, 15)

        drawer = ImageDraw.Draw(background)

        xml = XML(name='%05d' % i, size=(width, height))

        blocked_boxes = []

        for obj_ind in range(obj_quantity):
            label = random.choice(labels)

            font = random.choice(fonts)
            font_size = random.randint(int(0.1 / len(label) * min_w_h), int(2.5 / len(label) * min_w_h))
            font = ImageFont.truetype(font, size=font_size)

            text_size = font.getsize(label)

            curr_possible_x_y = create_x_y_from_blocks(
                expand_blocks(blocked_boxes, text_size),
                [width - text_size[0], height - text_size[1]]
            )

            try:
                left_x, up_y = random.choice(curr_possible_x_y)
            except:
                print('Exception:(possibly due to empty list, its OK)', sys.exc_info()[0])
                continue
            colour = random.choice(colours)

            try:
                drawer.text((left_x, up_y), label, font=font, fill=colour)
            except:
                drawer.text((left_x, up_y), label, font=font, fill=colour[0])
            
            blocked_boxes.append([left_x, up_y, left_x + text_size[0], up_y + text_size[1]])

            xml.add_object(name=label, coordinates=(left_x, up_y, left_x + text_size[0], up_y + text_size[1]))
            
        for fake_ind in range(fakes_quantity):
            label = random.choice(fakes)

            font = random.choice(fonts)
            font_size = random.randint(int(0.1 / len(label) * min_w_h), int(2.5 / len(label) * min_w_h))
            font = ImageFont.truetype(font, size=font_size)

            text_size = font.getsize(label)

            curr_possible_x_y = create_x_y_from_blocks(
                expand_blocks(blocked_boxes, text_size),
                [width - text_size[0], height - text_size[1]]
            )

            try:
                left_x, up_y = random.choice(curr_possible_x_y)
            except:
                print('Exception(possibly due to empty list, its OK)', sys.exc_info()[0])
                continue
            colour = random.choice(colours)

            try:
                drawer.text((left_x, up_y), label, font=font, fill=colour)
            except:
                drawer.text((left_x, up_y), label, font=font, fill=colour[0])

            blocked_boxes.append([left_x, up_y, left_x + text_size[0], up_y + text_size[1]])

        xml.save_file()
        try:
            background.save(SAVE_PATH + img_name)
        except:
            img_name = '%05d' % i + '.png'
            background.save(SAVE_PATH + img_name)

        t += time.time() - start

        if i > 0 and i % disp_step == 0:
            print('iteration ', i, 'done')
            print('average time per iteration: ', t / float(disp_step))
            t = 0

    print('Done')


def set_train_test(iterations, test_part=0.1):
    test_images = random.sample([i for i in range(iterations)], int(test_part * iterations))
    train_images = [i for i in range(iterations) if i not in test_images]

    with open(TRAIN_PATH + 'train.txt', 'w') as train_f:
        for i in train_images:
            train_f.write('%05d' % i + '\n')
    with open(TEST_PATH + 'test.txt', 'w') as test_f:
        for i in test_images:
            test_f.write('%05d' % i + '\n')


if __name__ == '__main__':
    iterations = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    if iterations > MAX_ITERATIONS:
        iterations = MAX_ITERATIONS
    last_saved = int(sys.argv[2])
    create_dataset(iterations=iterations, last_saved=last_saved)
    set_train_test(iterations + last_saved)
