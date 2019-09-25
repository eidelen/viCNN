""" This model trains a cnn to lable images containing crosses, squares and circles """

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn.functional as F
import time
import os

import Misc


class ConvNet(nn.Module):
    """Convolutional neural network - 2 conv layer, 3 fully connected"""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.fc1 = nn.Linear(96800, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2,2))
        x = x.view(-1, self.num_flat_features(x)) # to vector form
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x) -> int:
        """ Compute the number of remaining image pixels for one sample.
            Note: Code was taken out from a pytorch example
        """
        size = x.size()[1:]  # compute the number of remaining image pixels for one sample
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


print("pytorch version " + torch.__version__)
device = ""
if torch.cuda.is_available():
    device = "cuda:0"
    print( "cuda is on. %d GPUs" % (torch.cuda.device_count()) )
else:
    device = "cpu"
    print( "No cuda available" )
print("Use computational device: %s" % device)

# load data
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
trainingset_loader = torch.utils.data.DataLoader(trainingset,batch_size=9, shuffle=True,num_workers=8)

testset = datasets.ImageFolder(root='test', transform=test_transform)
testset_loader = torch.utils.data.DataLoader(testset,batch_size=9, shuffle=False,num_workers=8)

for cl in trainingset.classes:
    idx = trainingset.class_to_idx[cl]
    print("Class %s as idx %d" % (cl, idx))

n_classes = len(trainingset.classes)



model_file_name = "conv.pt"
if os.path.isfile(model_file_name):
    net = torch.load(model_file_name)
else:
    net = ConvNet()


net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.5)

for epoch in range(10000):

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
        if i % 5 == 4:
            print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss/5))
            running_loss = 0.0


    # test with testdata
    correct = 0
    total = 0
    wrongImgList = []
    with torch.no_grad():
        for data in testset_loader:
            images, labels = data
            out = net(images.to(device))
            _, predicted = torch.max(out.data, 1)
            total += labels.size(0)
            corrList = (predicted == labels.to(device))
            correct += corrList.sum().item()
            # copy wrong detected
            for c in range(corrList.shape[0]):
                if not corrList[c]:
                    wrongImgList.append(images[c])
                    #print( "Wrong classification of %s as %s: out signal %s" %
                    #       (trainingset.classes[labels[c]], trainingset.classes[predicted[c]], str(out[c])))

    runningtime = time.time() - start
    print('Network performance in eporch %d: %.2f  (%d,  %d)%%   -  time: %.1f ' % (epoch, 100 * correct / total , correct, total, runningtime))

    Misc.show_multiple_images(wrongImgList, 3)
    torch.save(net, model_file_name)

