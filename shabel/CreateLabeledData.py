
from PIL import Image, ImageDraw

def create_training_test( n_training, n_test, s ):


    img = Image.new("L", (s, s), color=60)
    d = ImageDraw.Draw(img)
    d.rectangle([10,10,80,80], fill=200)
    img.save('pil_red.png')


if __name__ == '__main__':
    create_training_test(4, 2, 256)