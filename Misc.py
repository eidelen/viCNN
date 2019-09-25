""" This module contains general code usable with model training """

import torch
from typing import List
import matplotlib.pyplot as plt

def do_label_matrix(l: torch.Tensor, nc: int) -> torch.Tensor:
    """
    This function transforms a label vector (classes 0 - n) into a label matrix,
    where the corresponding labeled class is 1.0 and other entries 0.0.
    :param l: Label vector
    :param nc: Overall number of classes
    :return: label matrix
    """
    b = l.shape[0]
    mat = torch.zeros(b, nc)
    for i in range(b):
        label = l[i]
        mat[i,label] = 1.0

    return mat


def show_multiple_images(images: List[torch.Tensor], nPerSide: int):
    """
    This function displays the given images in one figure.
    :param images: List of torch tensors
    :param nPerSide: How many images per side (2 -> 4, 3 -> 9)
    """
    plt.close('all')
    n = min(9, len(images))
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, n+1):
        img = images[i - 1] / 2.0 + 0.5
        npimg = img.numpy()
        fig.add_subplot(nPerSide, nPerSide, i)
        plt.imshow(npimg[0], cmap='gray', vmin=0, vmax=1)

    plt.show(block=False)
    plt.pause(1)