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
import time
import os
import copy
import pickle

from CommonTorchGlasses import get_evaluation_transform, get_training_transform, get_classes_file_path, get_model_path

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False # we dont want feature extraction layers to learn

if __name__ == '__main__':
    print("PyTorch Version: ", torch.__version__)
    print("Torchvision Version: ", torchvision.__version__)

    # use a pretrained model - if true, faster achieving good results
    use_pretrained_model = True
    # if false, each layer of the CNN is adapting. when true, only last layer is adapting
    do_only_feature_extraction = False
    # the batch size
    batch_size = 8
    # number of epochs
    num_epochs = 3
    # the learning rate
    learning_rate = 0.0006

    # image input size - cannot be changed
    image_input_size = 224

    # load training and validation data set
    training_set = datasets.ImageFolder(root='data/trainingData', transform=get_training_transform(image_input_size))
    validation_set = datasets.ImageFolder(root='data/validationData', transform=get_evaluation_transform(image_input_size))
    dataloaders = {'train': torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=0),
                    'val': torch.utils.data.DataLoader(validation_set, batch_size=batch_size, shuffle=True, num_workers=0)}

    # count samples per class
    number_of_classes = len(training_set.classes)
    training_count = torch.zeros(number_of_classes, dtype=torch.long)
    validation_count = torch.zeros(number_of_classes, dtype=torch.long)
    for _, target in training_set:
        training_count += target
    for _, target in validation_set:
        validation_count += target
    for cl in training_set.classes:
        idx = training_set.class_to_idx[cl]
        print("Training: Class %s as idx %d, n=%d" % (cl, idx,training_count[idx]))
    for cl in validation_set.classes:
        idx = validation_set.class_to_idx[cl]
        print("Validation: Class %s as idx %d, n=%d" % (cl, idx, validation_count[idx]))

    print("Saving classes and indices: %s" % (get_classes_file_path()))
    with open(get_classes_file_path(), "wb") as fp:
        class_items = [(training_set.class_to_idx[cl], cl) for cl in training_set.classes]
        pickle.dump(class_items, fp)

    # load existing model if available, otherwise make it from scratch
    if os.path.isfile(get_model_path()):
        print("Load existing model: %s" % (get_model_path()))
        model_ft = torch.load(get_model_path())
        model_ft.eval()
    else:
        # load pretrained resnet model
        print("Create model from scratch")
        model_ft = models.resnet18(pretrained=use_pretrained_model)
        # important to call this function before modifying the classification layer
        set_parameter_requires_grad(model_ft, do_only_feature_extraction)
        # replace classifier layer in the very end. instead of 1000 classes, we have just 2
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, number_of_classes)

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
            dl = dataloaders[phase]
            for step, (inputs, labels) in enumerate(dl):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model_ft(inputs)

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

    # load best model weights and save it
    model_ft.load_state_dict(best_model_wts)
    print("Saving model: %s" % (get_model_path()))
    torch.save(model_ft, get_model_path())

