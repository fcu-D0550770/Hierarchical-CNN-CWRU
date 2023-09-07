import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import time
import numpy as np
import shutil
import os
import argparse
from utils import progress_bar
from torchsummary import summary
from skimage import io
import cv2
from torch.utils.data import Dataset
from prettytable import PrettyTable

m = nn.Softmax()

cfg = {
    'root': [8, 'M', 16, 'M', 16, 'M', 'D'],  ##16,32,32
    'root1': [8, 'M', 16, 'M', 16, 'M'],  ##16,32,32
    '2': [16, 'M', 32, 'M'],##'D'
    '3': [16, 'M', 32, 'M', 32, 'D'],
    '4': [16, 32, 'M', 32, 32, 'M', 'D'],
    '5': [16, 32, 'M', 32, 32, 'M', 64, 'D'],
    '6': [16, 32, 32, 'M', 64, 64, 128, 'M', 'D'],
}

# 将所有的图片重新设置尺寸为64*64
w = 64
h = 64
c = 1


# 读取图片及其标签函数
def read_image(inNum):
    img_data_list = []
    labels = []
    n = 58
    count = 0
    if (inNum == 1):
        path = open(r'Your path/CWRU2D/train.txt')
    else:
        path = open(r'Your path/CWRU2D/test.txt')

    for li in path:
        # print("li = ",li)
        allImage = li.strip('\n')
        if 'FE' in allImage:
            count = count + 1
            n_findX = allImage.find('X')
            n_findc = allImage.find('.')
            path_tif = ("/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/CNC_Data_Image_orgin_9bit_label/" + allImage[
                n] + "/" + allImage[
                           n_findX:n_findc] + ".tif")
            input_img = io.imread(path_tif)
            input_img_resize = cv2.resize(input_img, (64, 64))
            img_data_list.append(input_img_resize)
            labels.append(li[n])
    img_data = np.array(img_data_list)
    print("count is here", count)
    #    print(img_data)
    img_data = img_data.astype('float32')
    img_label = np.array(labels)
    img_label = img_label.astype('int32')  # int64
    # print("img_label = ", img_label)
    return img_data, img_label


# train_data, train_label = read_image(1)  # 輸出FE train_data和train_label
# print("train_label", train_label)
# test_data, test_label = read_image(0)


class model(nn.Module):
    def __init__(self, size):
        super(model, self).__init__()
        self.features = self._make_layers(cfg[size])
        # self.features_down = self._make_layers(cfg_down[size], 16)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 4,bias=False)  ##4 classification
        )

    def forward(self, x):  ##define data pass then return
        # print("origin x: {}".format(x.shape))
        y = self.features(x)
        # y = self.features_down(x)
        x = y.view(y.size(0), -1)  ##define 1 dimension
        out = self.classifier(x)  ##then flatten
        return y, out

    def _make_layers(self, cfg, channels=1):
        layers = []
        in_channels = channels
        for x in cfg:
            print("x is here", x)
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]  ##initial in_channels=3
                print("channel_x", x)
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def model_one():
    return model('root1')


class mod_three(nn.Module):
    def __init__(self, size):
        super(mod_three, self).__init__()
        self.features = self._make_layers(cfg[size], 16)  # DEFAULT 32
        self.classifier = nn.Sequential(
            nn.Linear(32 * 2 * 2, 3,bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return x, out

    def _make_layers(self, cfg, channels=3):
        layers = []
        in_channels = channels
        for x in cfg:
            if x == 'D':
                layers += [nn.Dropout()]
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=False), nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def model_3():
    return mod_three('2')


def train_mod(model, optimizer, loss_fn, data, target, device):
    data = data.to(device)
    target = target.to(device)
    optimizer.zero_grad()
    _, net_out = model(data)
    loss = loss_fn(net_out, target)
    loss.backward()
    optimizer.step()
    pred = net_out.max(1, keepdim=True)[1]
    model.acc += pred.eq(target.view_as(pred)).sum().item()
    return loss


def train(model, model_three, optimizer, loss_fn, trainloader, valloader, device, maxi):
    max = maxi  ##maxi initial 0
    for epoch in range(30):
        model.eval()
        model_three.train()
        count = 0
        num_three = 0
        total = 0
        model_three.acc = 0
        for batch_num, (data, target) in enumerate(trainloader):
            data = data.to(device)
            target = target.to(device)
            target1 = target.clone().to(device)
            target1[(target1 == 0).nonzero()] = 100
            target1[(target1 == 1).nonzero()] = 101
            target1[(target1 == 2).nonzero()] = 101
            target1[(target1 == 3).nonzero()] = 101
            target1[(target1 == 4).nonzero()] = 102
            target1[(target1 == 5).nonzero()] = 102
            target1[(target1 == 6).nonzero()] = 102
            target1[(target1 == 7).nonzero()] = 103
            target1[(target1 == 8).nonzero()] = 103
            target1[(target1 == 9).nonzero()] = 103
            target1 -= 100  ##pick label 3,5,6,8 for 3

            optimizer.zero_grad()
            next_data, net_out = model(data)  ##16x8x8,4

            indices = (target1 == 2).nonzero()[:, 0]
            #print("indices here", indices)

            ##print("target1 is here",target1) classification label
            three_data = next_data[indices]
            three_target = target[indices]
            # print("three_data is here",three_data)
            # print("three_target is here",three_target)##only 3,5,6,8

            three_target1 = three_target.clone().to(device)
            #print("three_target1 here before = ", three_target1)

            three_target1[(three_target1 == 4).nonzero()] = 10
            three_target1[(three_target1 == 5).nonzero()] = 11
            three_target1[(three_target1 == 6).nonzero()] = 12
            three_target1 -= 10  ##final output 0,1,2
            #print("three_target1 here after = ", three_target1)

            loss = train_mod(model_three, optimizer, loss_fn, three_data, three_target1, device)
            num_three += three_data.shape[0]

            progress_bar(batch_num, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (loss / (batch_num + 1), 100. * model_three.acc / num_three, model_three.acc, num_three))
        print("Epoch: ", epoch + 1, "\nTrain: ", model_three.acc / num_three)

        #if model_three.acc / num_three > max:
         #   print("checkpoint saved")
          #  max = model_three.acc / num_three
           # torch.save(model_three.state_dict(), "CWRU_2.pth")
            # print("modelsavewhere",model_three.state_dict())

        model.eval()
        count1 = 0
        model_three.eval()
        model_three.acc = 0
        num_three = 0
        count_three = 0
        for data, target in valloader:
            data = data.to(device)
            target = target.to(device)
            target1 = target.clone().to(device)
            target1[(target1 == 0).nonzero()] = 100
            target1[(target1 == 1).nonzero()] = 101
            target1[(target1 == 2).nonzero()] = 101
            target1[(target1 == 3).nonzero()] = 101
            target1[(target1 == 4).nonzero()] = 102
            target1[(target1 == 5).nonzero()] = 102
            target1[(target1 == 6).nonzero()] = 102
            target1[(target1 == 7).nonzero()] = 103
            target1[(target1 == 8).nonzero()] = 103
            target1[(target1 == 9).nonzero()] = 103
            target1 -= 100
            next_data, net_out = model(data)

            indices = (target1 == 2).nonzero()[:, 0]

            three_data = next_data[indices]
            three_target = target[indices]

            three_target1 = three_target.clone().to(device)
            three_target1[(three_target1 == 4).nonzero()] = 10
            three_target1[(three_target1 == 5).nonzero()] = 11
            three_target1[(three_target1 == 6).nonzero()] = 12
            three_target1 -= 10

            _, three_out = model_three(three_data)
            pred_three = three_out.max(1, keepdim=True)[1]
            count_three += pred_three.eq(three_target1.view_as(pred_three)).sum().item()
            num_three += three_data.shape[0]

        print("Val: ", count_three / num_three)

        if count_three / num_three > max:
            print("checkpoint saved")
            max = count_three / num_three
            torch.save(model_three.state_dict(), "CWRU_2.pth")
            # print("modelsavewhere",model_three.state_dict())

    return max


def average_softmax(model, trainloader, valloader, device):
    nb_classes = 10
    out_classes = 10
    counts = [0 for i in range(nb_classes)]
    soft_out = torch.zeros((nb_classes, 1, nb_classes)).cuda()
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(valloader):
            inputs = inputs.to(device)
            classes = classes.to(device)
            _, outputs = model(inputs)
            outputs = m(outputs)
            for categ in range(nb_classes):
                indices = (classes == categ).nonzero()[:, 0]
                hold = outputs[indices]
                soft_out[categ] += hold.sum(dim=0)
                counts[categ] += hold.shape[0]
    for i in range(nb_classes):
        soft_out[i] = soft_out[i] / counts[i]
    print(soft_out)


class MyDataset(Dataset):
    def __init__(self, txt):
        fh = open(txt, 'r')
        img_data_list = []
        label = []

        for line in fh:
            allImage = line.strip('\n')
            if 'FE' in allImage:
                path = '/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/CNC_Data_Image_orgin_9bit_label/'
                path_new = path + line.split('/')[-2] + '/' + line.split('/')[-1].split('.')[-2] + '.tif'
                input_img = io.imread(path_new)
                input_img_resize = cv2.resize(input_img, (64, 64))
                input_img_resize = np.resize(input_img_resize, (1, 64, 64))
                label_new = line.split('/')[5]
                img_data_list.append(input_img_resize)
                label.append(int(label_new))

        self.img = img_data_list
        self.label = label

    def __getitem__(self, idx):

        # img = self.img[idx]
        # # img = img.astype('float32')
        # label = self.label[idx]
        # label = np.array(label)
        return self.img[idx], self.label[idx]

    def __len__(self):
        return self.img.__len__()


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


if __name__ == "__main__":

    train_dataset = MyDataset(txt='/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/train.txt')

    val_dataset = MyDataset(txt='/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/test.txt')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=None,  ##no shuffle
                                               batch_size=64)  ##train_set have something problem

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             sampler=None,  ##no shuffle
                                             batch_size=64)  ##test_set

    train_data, train_label = read_image(1)  # 輸出FE train_data和train_label
    print("train_label", train_label)

    model = model_one().to(torch.device("cuda"))  ##root
    model.load_state_dict(torch.load('./CWRU_root.pth'))
    model_three = model_3().to(torch.device("cuda"))

    model_three.load_state_dict(torch.load('./CWRU_2.pth'))


    # print(summary(model, (1, 64, 64)))

    print(summary(model_three, (16, 8, 8)))

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model_three.parameters(), lr=learning_rate, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    # print("stop here now")
    count_parameters(model)
    max = train(model, model_three, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), 0)

    for i in range(3):
        #model.load_state_dict(torch.load('./model_saved/cwru_root.pth'))
        #print("i is here", i)
        model_three.load_state_dict(torch.load('./CWRU_2.pth'))

        learning_rate /= 10
        optimizer = torch.optim.Adam(model_three.parameters(), lr=learning_rate, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        max = train(model, model_three, optimizer, loss_fn, train_loader, val_loader, torch.device("cuda"), max)
    #model.load_state_dict(torch.load('./CWRU_2.pth'))
    model_three.load_state_dict(torch.load('./CWRU_2.pth'))

    average_softmax(model, train_loader, val_loader, torch.device("cuda"))
