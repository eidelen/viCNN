""" This model trains a cnn to lable images containing crosses, squares and circles """

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import time
import os

import Misc
import shabel.ShabelModel as ShabelM


def test_all_classes(data_loader: torch.utils.data.DataLoader, modelNet: nn.Module, torch_dev: str):

    data_set = data_loader.dataset
    nbr_classes = len(data_set.classes)

    total_class = [0] * nbr_classes
    total_class_correct = [0] * nbr_classes
    total_correct = 0
    total = 0
    wrong_sample_list = []


    with torch.no_grad():
        for data in data_loader:
            img, label = data
            out = modelNet(img.to(torch_dev))
            _, predicted = torch.max(out.data, 1)
            total += label.size(0)
            corrList = (predicted == label.to(torch_dev))
            total_correct += corrList.sum().item()

            # copy wrong detected
            for c in range(corrList.shape[0]):

                cl_idx = label[c]
                total_class[cl_idx] = total_class[cl_idx] + 1.0

                if corrList[c]:
                    total_class_correct[cl_idx] = total_class_correct[cl_idx] + 1.0
                else:
                    wrong_sample_list.append(img[c])

    return total, total_correct, wrong_sample_list, total_class, total_class_correct



def do_training(learning_rate: float, batch_size: int, gui: False, nbr_epochs: int ):

    device = Misc.get_torch_device()
    print("Use computational device: %s" % device)

    # load data & augmentation
    training_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                         transforms.RandomCrop(232),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5, )) ])

    test_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                         transforms.Resize(232),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5,), (0.5, )) ])

    trainingset = datasets.ImageFolder(root='training', transform=training_transform)
    trainingset_loader = torch.utils.data.DataLoader(trainingset,batch_size=batch_size, shuffle=True,num_workers=8)

    testset = datasets.ImageFolder(root='test', transform=test_transform)
    testset_loader = torch.utils.data.DataLoader(testset,batch_size=1, shuffle=False,num_workers=1)

    for cl in trainingset.classes:
        idx = trainingset.class_to_idx[cl]
        print("Class %s as idx %d" % (cl, idx))

    n_classes = len(trainingset.classes)

    model_file_name = "conv.pt"
    if os.path.isfile(model_file_name):
        net = torch.load(model_file_name)
        net.eval()
    else:
        net = ShabelM.ConvNet()


    net.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    best_performance = 0.0
    for epoch in range(nbr_epochs):

        running_loss = 0.0
        start = time.time()

        for i, batch in enumerate(trainingset_loader,0):
            x, y = batch
            y = Misc.do_label_matrix(y, n_classes).to(device)

            optimizer.zero_grad()
            out = net(x.to(device))

            loss = criterion(out,y)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 15 ==14:
                print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss/15))
                running_loss = 0.0


        # test with testdata
        total, correct, wrong_img_list, total_class, total_class_correct = test_all_classes(testset_loader, net, device)
        performance = 100.0 * correct / total

        if best_performance < performance:
            best_performance = performance

        if gui:
            Misc.show_multiple_images(wrong_img_list, 3)

        runningtime = time.time() - start
        print('Network performance in epoch %d: %.2f  (%d,  %d)%%   -  time: %.1f ' % (epoch, performance, correct, total, runningtime))
        for cl_idx in range(len(total_class)):
            cl_name = testset.classes[cl_idx]
            cl_performance = 100.0 * total_class_correct[cl_idx] / total_class[cl_idx]
            print('%s: %.2f (%d, %d)' % (cl_name, cl_performance, total_class_correct[cl_idx], total_class[cl_idx]) )

        torch.save(net, model_file_name)


    return best_performance



def find_hypers(lrs, bss, epochs):
    Misc.print_torch_info()

    best_hypers = (0.0, 0.0, 0.0)

    for lr, bs in [(i, j) for i in lrs for j in bss]:
        print("CHECK HYPER (%.4f, %.4f)" % (lr, bs))
        performance = do_training(learning_rate=lr, batch_size=bs, gui=False, nbr_epochs=epochs)

        if performance > best_hypers[0]:
            best_hypers = (performance, lr, bs)

        print("PERFORMANCE %.2f (%4f, %4f)" % (performance, lr, bs))

    print("Best Hpyerparams -> ", best_hypers)



if __name__ == '__main__':
    lrs = [0.015]
    bss = [3]

    find_hypers(lrs, bss, 300)


