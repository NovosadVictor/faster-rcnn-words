from xml_class import XML
import numpy as np
from PIL import Image


RAZMETKA_FILE = 'razmetka.txt'
IMAGES_PATH = 'data/real_imgs/'

def create_xml():
    with open(RAZMETKA_FILE, 'r') as f:
        lines = f.readlines()
        for ind, img in enumerate(lines):
            img = img.split()
            image_name = img[0] + '.jpg'
            image = Image.open(IMAGES_PATH + image_name)
            width, height = image.size
            boxes = img[1:]
            boxes = np.reshape(boxes, (-1, 5)).tolist()
            for i in range(len(boxes)):
                for j in range(len(boxes[i]) - 1):
                    boxes[i][j] = float(boxes[i][j])
            xml = XML(name='real_%d' % ind, size=(width, height))
            for box in boxes:
                xml.add_object(name=box[4], coordinates=(box[0], box[1], box[2], box[3]))

            xml.save_file()


if __name__ == '__main__':
    create_xml()