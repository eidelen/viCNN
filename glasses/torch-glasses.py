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

plt.ion()   # interactive mode


class GlassesVottDataSet(Dataset):
    """Glasses dataset from VOTT labeling tool."""

    def __init__(self, csv_file, transform=None):
        """
        Dataset representing the glasses labeled by the VOTT labeling software
        @param csv_file: Path to csv file
        @param transform: Optional transform
        """
        self.transform = transform
        self.label_file_list = []
        label_list = []
        parent_dir = Path(csv_file).parent.absolute()
        with open(csv_file, newline='') as csvfile:
            labeling_input = csv.reader(csvfile, delimiter=',', quotechar='|')
            labeling_input_iter = iter(labeling_input)
            next(labeling_input_iter)
            for row in labeling_input_iter:
                file_name = (row[0])[1:-1]
                file_path_abs = str(parent_dir / file_name)
                text_label = (row[5])[1:-1]

                if text_label not in label_list:
                    label_list.append(text_label)
                num_label = label_list.index(text_label)

                self.label_file_list.append((num_label, text_label, file_path_abs))
        self.num_classes = len(label_list)

    def __len__(self):
        return len(self.label_file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file_path = self.label_file_list[idx][2]
        image = io.imread(img_file_path)

        class_id = self.label_file_list[idx][0]
        label_vect = np.zeros((self.num_classes, 1))
        label_vect[class_id, 0] = 1.0
        label_vect = label_vect.astype('float')

        sample = {'image': image, 'label': label_vect}

        if self.transform:
            sample = self.transform(sample)

        return sample


def load_labeled_data(vott_csv_file_path: str):

    labeled_data = []
    label_list = []

    parent_dir = Path(vott_csv_file_path).parent.absolute()
    with open(vott_csv_file_path, newline='') as csvfile:
        labeling_input = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in labeling_input:
            file_name = (row[0])[1:-1]
            file_path_abs = str(parent_dir / file_name)
            text_label = (row[5])[1:-1]

            if text_label not in label_list:
                label_list.append(text_label)

            num_label = label_list.index(text_label)

            labeled_data.append( (num_label, text_label, file_path_abs) )

    return labeled_data


def show_sample(image, label):
    plt.imshow(image)
    print(label)
    print(image.shape, label.shape)
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

    #labeled_data = load_labeled_data('labeledData/vott-csv-export/Glasses-export.csv')

    glasses_dataset = GlassesVottDataSet(csv_file='labeledData/vott-csv-export/Glasses-export.csv')
    show_all_samples(glasses_dataset)


