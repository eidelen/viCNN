
from PIL import Image, ImageDraw
import random
import os
import shutil
import numpy as np
from typing import  Tuple

def get_random_square(img_size) -> Image:
    """
    Create a random gray scale square image.
    :param img_size: image size
    :return: Image
    """
    img = Image.new("L", (img_size, img_size), color=0)
    d = ImageDraw.Draw(img)
    sq_side = random.randint(16, img_size/4)
    start_x = img_size / 2 - sq_side
    start_y = start_x
    d.rectangle([start_x, start_y, start_x+sq_side, start_y+sq_side], fill=random.randint(80,255))
    q = img.rotate(random.randint(0,360), resample=Image.BICUBIC )
    return q


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


def add_gaussian_noise(img: Image, var: float) -> Image:
    """
    Add Gaussian noise to an image.
    :param img: image
    :param var: Variance
    :return: Image
    """
    col, row = img.size
    mean = 0
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noise_img = gauss + np.array(img)
    np.place( noise_img, noise_img < 0.0, 0.0 )
    np.place( noise_img, noise_img > 255.0, 255.0)
    return Image.fromarray( noise_img.astype(np.uint8) )


def create_training_test( n_training: int, n_test: int, s: int, noise: float ):
    """
    Creates training and test data for artificial squares and circles. The files are saved under
    test/xxx/ and training/xxx/
    :param n_training: number of training samples
    :param n_test: number of test samples
    :param s: image size
    :param noise: image noise (Gaussian variance)
    :return: None
    """

    shutil.rmtree("training", True)
    shutil.rmtree("test", True)

    sq_training = "training/square/"
    sq_test = "test/square/"
    os.makedirs(sq_training)
    os.makedirs(sq_test)
    for i in range(0, n_training):
        img = get_random_square(s)
        n_img = add_gaussian_noise(img=img, var=noise)
        n_img.save(sq_training + str(i) +".png")
    for i in range(0, n_test):
        img = get_random_square(s)
        n_img = add_gaussian_noise(img=img, var=noise)
        n_img.save(sq_test + str(i) +".png")

    sq_training = "training/circle/"
    sq_test = "test/circle/"
    os.makedirs(sq_training)
    os.makedirs(sq_test)
    for i in range(0, n_training):
        img = get_random_circle(s)
        n_img = add_gaussian_noise(img=img, var=noise)
        n_img.save(sq_training + str(i) + ".png")
    for i in range(0, n_test):
        img = get_random_circle(s)
        n_img = add_gaussian_noise(img=img, var=noise)
        n_img.save(sq_test + str(i) + ".png")


if __name__ == '__main__':
    create_training_test(1000, 100, 256, 4.0)