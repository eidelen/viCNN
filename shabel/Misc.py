import torch


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