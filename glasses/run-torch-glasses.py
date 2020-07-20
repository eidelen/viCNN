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

    # load image
    path_to_test_image = 'labeling/validationData/vott-csv-export/glasses_on_1.mov#t=2.2.jpg'
    image = Image.open(path_to_test_image)
    image_t = t(image).unsqueeze_(0)
    image_t = image_t.to(device)

    # load model
    print("Load model: %s" % (res_model_path))
    model_ft = torch.load(res_model_path)
    model_ft.eval()
    model_ft = model_ft.to(device)

    # classification
    prediction = model_ft(image_t)
    class_idx = (torch.max(prediction, 1)[1]).data.cpu().numpy()[0]
    class_str = classes[class_idx][1]
    print(class_str)

    #cap = cv2.VideoCapture('labeling/input/basel_adi_noglasses.mov')
    cap = cv2.VideoCapture(0)
    while (True):
        ret, frame = cap.read()

        pil_image = Image.fromarray(frame)
        image_t = t(pil_image).unsqueeze_(0)
        image_t = image_t.to(device)

        prediction = model_ft(image_t)
        class_idx = (torch.max(prediction, 1)[1]).data.cpu().numpy()[0]
        class_str = classes[class_idx][1]
        print(class_str, prediction)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
