
from PIL import Image, ImageDraw
import random
import os
import shutil
from typing import  Tuple

def get_random_square(img_size) -> Image:
    """
    Create a random gray scale square image.
    :param img_size: image size
    :return: Image
    """
    img = Image.new("L", (img_size, img_size), color=0)
    d = ImageDraw.Draw(img)

    sq_side = random.randint(10, img_size/4)
    start_x = img_size / 2 - sq_side
    start_y = start_x

    d.rectangle([start_x, start_y, start_x+sq_side, start_y+sq_side], fill=random.randint(80,255))

    return img.rotate(random.randint(0,360), resample=Image.BILINEAR )


def get_random_circle(img_size) -> Image:
    """
    Create a random gray scale circle image.
    :param img_size: image size
    :return: Image
    """
    img = Image.new("L", (img_size, img_size), color=0)
    d = ImageDraw.Draw(img)

    radius = random.randint(10, img_size/4)
    start_x = img_size / 2 - random.randint(-img_size / 4, img_size / 4)
    start_y = img_size / 2 - random.randint(-img_size / 4, img_size / 4)

    d.ellipse([start_x-radius, start_y-radius, start_x+radius, start_y+radius], fill=random.randint(80,255))

    return img


def create_training_test( n_training, n_test, s ):

    shutil.rmtree("training", True)
    shutil.rmtree("test", True)

    sq_training = "training/square/"
    sq_test = "test/square/"
    os.makedirs(sq_training)
    os.makedirs(sq_test)
    for i in range(0, n_training):
        img = get_random_square(s)
        img.save(sq_training + str(i) +".png")
    for i in range(0, n_test):
        img = get_random_square(s)
        img.save(sq_test + str(i) +".png")

    sq_training = "training/circle/"
    sq_test = "test/circle/"
    os.makedirs(sq_training)
    os.makedirs(sq_test)
    for i in range(0, n_training):
        img = get_random_circle(s)
        img.save(sq_training + str(i) + ".png")
    for i in range(0, n_test):
        img = get_random_circle(s)
        img.save(sq_test + str(i) + ".png")




if __name__ == '__main__':
    create_training_test(100, 50, 256)