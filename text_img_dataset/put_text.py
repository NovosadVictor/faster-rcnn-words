import PIL
from PIL import ImageFont, ImageDraw, Image
from os import listdir  # list of files
import random
import sys
from xml_class import XML  # custom xml


BACKGROUNDS_PATH = 'backgrounds/'
FONTS_PATH = 'fonts/'
SAVE_PATH = 'data/Images/'
TRAIN_PATH = 'data/ImageSets/'
TEST_PATH = 'data/ImageSets/'
FAKE_WORDS_PATH = './fakewords.txt'
MAX_ITERATIONS = 10000

labels = ['Aeroexpress', 'Microsoft', 'Apple', ]


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


fakes = load_fake_words()

fonts = [FONTS_PATH + file for file in listdir(FONTS_PATH)]  # fonts path

backgrounds = [BACKGROUNDS_PATH + file for file in listdir(BACKGROUNDS_PATH)]  # files path

random.seed(1)

colours = [(r, g, b)
           for r in range(0, 256)
           for g in range(0, 256)
           for b in range(0, 256)]

all_images = []


def create_dataset(iterations, disp_step=100):

    for i in range(iterations):
        img_name = '%05d' % i + '.jpg'
        background = random.choice(backgrounds)
        background = Image.open(background)
      
        width, height = background.size  # image size
        min_w_h = min(width, height)
        obj_quantity = random.randint(0, 4)
        fakes_quantity = random.randint(0, 20)

        drawer = ImageDraw.Draw(background)

        xml = XML(name='%05d' % i, size=(width, height))

        possible_x_y = [(x, y)
                        for x in range(1, width)
                        for y in range(1, height)]
        prev_box = [width, height, width, height]

        for obj_ind in range(obj_quantity):
            font = random.choice(fonts)
            font_size = random.randint(int(0.04 * min_w_h), int(0.3 * min_w_h))
            font = ImageFont.truetype(font, size=font_size)

            label = random.choice(labels)

            text_size = font.getsize(label)

            curr_possible_x_y = [(x, y)
                                 for (x, y) in possible_x_y
                                     if (x < width - text_size[0] and x < prev_box[0] - text_size[0]
                                     and y < height - text_size[1]]

            try:
                left_x, up_y = random.choice(curr_possible_x_y)
            except:
                continue
            colour = random.choice(colours)

            try:
                drawer.text((left_x, up_y), label, font=font, fill=colour)
            except:
                drawer.text((left_x, up_y), label, font=font, fill=colour[0])
            
            prev_box = [left_x, up_y, left_x + text_size[0], up_y + text_size[1]]
            possible_x_y = [(x, y) for (x, y) in possible_x_y
                                 if (x < prev_box[0] or x > prev_box[2])
                                 and (y < prev_box[1] or y > prev_box[3])]

            xml.add_object(name=label, coordinates=(prev_box[0], prev_box[1], prev_box[2], prev_box[3]))
            

        for fake_ind in range(fakes_quantity):
            font = random.choice(fonts)
            font_size = random.randint(int(0.02 * min_w_h), int(0.3 * min_w_h))
            font = ImageFont.truetype(font, size=font_size)

            label = random.choice(fakes)

            text_size = font.getsize(label)

            curr_possible_x_y = [(x, y)
                                 for (x, y) in possible_x_y
                                     if x < width - text_size[0]
                                     and y < height - text_size[1]]

            try:
                left_x, up_y = random.choice(curr_possible_x_y)
            except:
                continue
            colour = random.choice(colours)

            try:
                drawer.text((left_x, up_y), label, font=font, fill=colour)
            except:
                drawer.text((left_x, up_y), label, font=font, fill=colour[0])

            possible_x_y = [(x, y) for (x, y) in possible_x_y
                            if (x < left_x or x > left_x + text_size[0])
                            and (y < up_y or y > up_y + text_size[1])]
        
        xml.save_file()

        background.save(SAVE_PATH + img_name)

        if i > 0 and i % disp_step == 0:
            print('iteration ', i, 'done')

    print('Done')


def set_train_test(iterations, test_part=0.25):
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
    create_dataset(iterations)
    set_train_test(iterations)
