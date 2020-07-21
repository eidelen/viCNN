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
import pickle

from CommonTorchGlasses import VottImageClassDataSet, get_evaluation_transform, get_training_transform

plt.ion()   # interactive mode

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # we dont want feature extraction layers to learn

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)
    res_model_path = 'resnet_pytorch.pt'
    class_file_path = 'classes.txt'

    use_pretrained_model = True
    do_only_feature_extraction = True # we dont want the front part of the network to change
    batch_size = 16
    num_epochs = 30
    learning_rate = 0.0003

    image_input_size = 224

    # load training and validation data set
    training_set = VottImageClassDataSet(csv_file='data/trainingData/vott-csv-export/glasses_training-export.csv', transform=get_training_transform(image_input_size))
    print("Number of training samples: %d" % (len(training_set)))
    validation_set = VottImageClassDataSet(csv_file='data/validationData/vott-csv-export/glasses_training-export.csv', transform=get_evaluation_transform(image_input_size))
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


    optimizer = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9)
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
    print("Saving classes: %s" % (class_file_path))
    with open(class_file_path, "wb") as fp:
        pickle.dump(training_set.get_classes(), fp)
