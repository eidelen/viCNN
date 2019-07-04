import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import time


class ConvNet(nn.Module):
    """Convolutional neural network - 2 conv layer, 3 fully connected"""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, 5)
        self.conv2 = nn.Conv2d(256, 64, 5)
        self.fc1 = nn.Linear(238144, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
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


def showMultipleImages(imgs):
    plt.close('all')
    n = min( 9, len(imgs) )
    fig = plt.figure(figsize=(10, 10))
    for i in range(1, n+1):
        img = imgs[i - 1] / 2.0 + 0.5
        npimg = img.numpy()
        fig.add_subplot(3, 3, i)
        plt.imshow(npimg[0], cmap='gray', vmin=0, vmax=1)

    plt.show(block=False)
    plt.pause(1)


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


print("pyTorch version" + torch.__version__)
device = ""
if torch.cuda.is_available():
    device = "cuda:0"
    print( "Cuda is on. %d GPUs" % (torch.cuda.device_count()) )
else:
    device = "cpu"
    print( "No Cuda" )
print("Use computational device: %s" % device)

# load data
data_transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, )) ])

trainingset = datasets.ImageFolder(root='training', transform=data_transform)
trainingset_loader = torch.utils.data.DataLoader(trainingset,batch_size=32, shuffle=True,num_workers=8)

testset = datasets.ImageFolder(root='test', transform=data_transform)
testset_loader = torch.utils.data.DataLoader(testset,batch_size=16, shuffle=False,num_workers=8)

for cl in trainingset.classes:
    idx = trainingset.class_to_idx[cl]
    print("Class %s as idx %d" % (cl, idx))

n_classes = len(trainingset.classes)


# training
net = ConvNet()

if torch.cuda.device_count() > 1 and False: #GPU parallelisation
    net = nn.DataParallel(net.cuda())

net.to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.5)

for epoch in range(10000):

    running_loss = 0.0
    wrongImgList = []
    start = time.time()

    for i, batch in enumerate(trainingset_loader,0):
        x, y = batch
        y = do_label_matrix(y,n_classes).to(device)

        optimizer.zero_grad()
        out = net(x.to(device))

        loss = criterion(out,y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.8f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


    # test with testdata
    correct = 0
    total = 0
    wrongImgList.clear()
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
                    print( "Wrong class %d: out signal %s" % (labels[c], str(out[c])))

    runningtime = time.time() - start
    print('Accuracy of the network on test images: %.2f  (%d,  %d)%%   -  time: %.1f ' % (
            100 * correct / total , correct, total, runningtime))

    showMultipleImages(wrongImgList)

