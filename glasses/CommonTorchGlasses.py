from __future__ import print_function
from __future__ import division
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import cv2

def get_training_transform(image_input_size):
    """
    Get the pytorch training transformations supporting data augmentation.
    @param image_input_size: target image size.
    @return: tensor ready to enter neural network
    """
    return transforms.Compose([
            transforms.RandomResizedCrop(image_input_size, scale=(0.8, 1.2), ratio=(0.9, 1.1)),
            transforms.RandomRotation((-20,20)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def get_evaluation_transform(image_input_size):
    """
    Get the pytorch evalutation transformations.
    @param image_input_size: target image size.
    @return: tensor ready to enter neural network
    """
    return transforms.Compose([
            transforms.Resize(image_input_size),
            transforms.CenterCrop(image_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


def show_sample(image, label):
    plt.imshow(image)
    print(label)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_all_samples(dataset):
    _ = plt.figure()
    for i in range(len(dataset)):
        sample = dataset[i]
        show_sample(**sample)
        plt.show()


class CvFaceCapture:
    """ This class detects faces in an video stream by using opencv. """

    def __init__(self, capture):
        """
        @param capture: Open cv video capture object
        """
        self.face_detector = cv2.CascadeClassifier(
            '/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
        self.capture = capture

    def read(self):
        """ Reads the next video frame and detects the faces on it. """
        ret, frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        return frame, faces

