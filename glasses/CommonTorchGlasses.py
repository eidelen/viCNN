from __future__ import print_function
from __future__ import division
import torch
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
import numpy as np
import csv
from pathlib import Path
from PIL import Image
from PIL import Image
import matplotlib.pyplot as plt
from typing import List, Tuple

class VottImageClassDataSet(Dataset):
    """Image / 1 class label dataset labeled with VOTT labeling tool (exported as CSV)."""

    def __init__(self, csv_file, transform=None):
        """
        Dataset labeled by the VOTT labeling software
        @param csv_file: Path to csv file
        @param transform: Optional transform
        """
        self.transform = transform
        path_label = []
        self.label_list = []
        parent_dir = Path(csv_file).parent.absolute()

        with open(csv_file, newline='') as csvfile:
            labeling_input = csv.reader(csvfile, delimiter=',', quotechar='|')
            labeling_input_iter = iter(labeling_input)
            next(labeling_input_iter)
            for row in labeling_input_iter:
                file_name = (row[0])[1:-1]
                file_path_abs = str(parent_dir / file_name)
                text_label = (row[5])[1:-1]

                if text_label not in self.label_list:
                    self.label_list.append(text_label)
                path_label.append((text_label, file_path_abs))

        # set a label index
        self.label_list.sort()
        self.label_file_list = []
        for class_text, file_path in path_label:
            class_idx = self.label_list.index(class_text)
            self.label_file_list.append( (class_idx, class_text, file_path) )

        self.num_classes = len(self.label_list)

    def __len__(self) -> int:
        return len(self.label_file_list)

    def get_classes(self) -> List[Tuple[int, str]]:
        """
        Returns the list of classes and their indices. The list
        is created during parsing of cvs file.
        @return: list of label classes
        """
        return [(i, self.label_list[i]) for i in range(0, len(self.label_list))]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_file_path = self.label_file_list[idx][2]
        image = Image.open(img_file_path)

        class_id = self.label_file_list[idx][0]
        label_vect = np.zeros((self.num_classes))
        label_vect[class_id] = 1
        label_vect = label_vect.astype('long')

        # label does not require any transformation.
        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'label': label_vect}

        return sample


def get_training_transform(image_input_size):
    """
    Get the pytorch training transformations supporting data augmentation.
    @param image_input_size: target image size.
    @return: tensor ready to enter neural network
    """
    return transforms.Compose([
            transforms.RandomResizedCrop(image_input_size, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
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
    fig = plt.figure()
    for i in range(len(dataset)):
        sample = dataset[i]
        show_sample(**sample)
        plt.show()