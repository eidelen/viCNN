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

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # we dont want feature extraction layers to learn


if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    res_model_path = 'resnet_pytorch.pt'


    use_pretrained_model = True
    do_only_feature_extraction = True # we dont want the front part of the network to change
    batch_size = 12
    num_epochs = 3

    image_input_size = 224

    data_transforms = {
    # Data augmentation and normalization for training
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_input_size, scale=(0.8, 1.2), ratio=(0.8, 1.2)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    # Just normalization for validation
        'val': transforms.Compose([
            transforms.Resize(image_input_size),
            transforms.CenterCrop(image_input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'show': transforms.Compose([
            transforms.RandomResizedCrop(image_input_size, scale=(0.8, 1.2),ratio=(0.8, 1.2) ),
            transforms.RandomHorizontalFlip()
        ]),
    }

    #show_set =  GlassesVottDataSet(csv_file='labeling/trainingData/vott-csv-export/glasses_training-export.csv', transform=data_transforms['show'])
    #show_all_samples(show_set)

    # load training and validation data set
    training_set = GlassesVottDataSet(csv_file='labeling/trainingData/vott-csv-export/glasses_training-export.csv', transform=data_transforms['train'])
    print("Number of training samples: %d" % (len(training_set)))
    validation_set = GlassesVottDataSet(csv_file='labeling/validationData/vott-csv-export/glasses_training-export.csv', transform=data_transforms['val'])
    print("Number of validation samples: %d" % (len(validation_set)))
    num_classes = training_set.num_classes
    dataloaders = {'train': torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=4),
                        'val': torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=4)}

    # load existing model if available, otherwise make it from scratch
    if os.path.isfile(res_model_path):
        print("Load existing model: %s" % (res_model_path))
        model_ft = torch.load(res_model_path)
        model_ft.eval()
    else:
        # load pretrained resnet model
        print("Create model from scratch")
        model_ft = models.resnet18(pretrained=use_pretrained_model)
        # important to call this function before modifying the classification layer
        set_parameter_requires_grad(model_ft, do_only_feature_extraction)
        # replace classifier layer in the very end. instead of 1000 classes, we have just 2
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    # detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Computation device: %s" % (device))

    # print part of the model which requires training
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if do_only_feature_extraction:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)


    optimizer = optim.SGD(params_to_update, lr=0.0005, momentum=0.9)
    criterion = nn.CrossEntropyLoss()


    # do training and validation
    since = time.time()

    val_acc_history = []
    best_model_wts = copy.deepcopy(model_ft.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model_ft.train()  # Set model to training mode
            else:
                model_ft.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # accumulate over data
            for a_batch in dataloaders[phase]:
                inputs = a_batch['image']
                labels = a_batch['label']
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)

                    labels = torch.max(labels, 1)[1]
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_ft.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model_ft.load_state_dict(best_model_wts)
    print("Saving model: %s" % (res_model_path))
    torch.save(model_ft, res_model_path)
    print("Saving done")
