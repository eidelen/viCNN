"""
This code was inspired by
https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
"""

from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
from pathlib import Path
from skimage import io, transform
from PIL import Image
import pickle
import cv2
import pafy

plt.ion()  # interactive mode


def show_sample(image, label):
    plt.imshow(image)
    print(label)
    plt.pause(0.001)  # pause a bit so that plots are updated


def show_all_samples(dataset):
    fig = plt.figure()
    for i in range(len(dataset)):
        sample = dataset[i]
        show_sample(**sample)
        plt.show()


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Computation device: %s" % (device))

    res_model_path = 'resnet_pytorch.pt'
    class_file_path = 'classes.txt'

    # load classes (idx, strings)
    with open(class_file_path, "rb") as fp:
        classes = pickle.load(fp)

    # image input transformation
    t = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


    # load model
    print("Load model: %s" % (res_model_path))
    model_ft = torch.load(res_model_path)
    model_ft.eval()
    model_ft = model_ft.to(device)

    face_cascade = cv2.CascadeClassifier(
        '/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')

    url = 'https://www.youtube.com/watch?v=ZJPQBBIWBl8'
    vPafy = pafy.new(url)
    play = vPafy.getbestvideo(preftype="webm")
    cap = cv2.VideoCapture(play.url)

    #cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()
        if not ret:
            continue

        # find faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for i in range(0, len(faces)):
            [x, y, w, h] = faces[i]
            face = frame[y:y + h, x:x + w]
            pil_image = Image.fromarray(face)
            image_t = t(pil_image).unsqueeze_(0)
            image_t = image_t.to(device)

            prediction = model_ft(image_t)
            class_idx = (torch.max(prediction, 1)[1]).data.cpu().numpy()[0]
            class_str = classes[class_idx][1]
            print(class_str, prediction)

            pen_color = (0, 0, 255) if class_idx == 0 else (0, 255, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), pen_color, 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, 'Class: ' + class_str, (40, 40 + 50 * i), font, 1, pen_color, 2, cv2.LINE_AA)

        cv2.imshow('frame', frame)

        keyCode = cv2.waitKey(1)
        if keyCode == ord('q'):
            break;


    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
