from __future__ import print_function
from __future__ import division
from torchvision import transforms

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

def get_classes_file_path() -> str:
    """
    Get the path to the file which defines the classes and their corresponding indices.
    @return: file path
    """
    return 'torch_classes.txt'

def get_model_path() -> str:
    """
    Get model torch model file path.
    @return: file path
    """
    return 'torch_model.pt'
