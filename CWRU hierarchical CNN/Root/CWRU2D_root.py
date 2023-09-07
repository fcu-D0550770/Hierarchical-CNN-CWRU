import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

import math
from torchvision import datasets, transforms
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
from torch.quantization import QuantStub, DeQuantStub
import torch.quantization

from binarizers import WeightBinarizer, ActivationBinarizer, Ternarizer, Identity, Sign
from utils1 import _pair

from quant_dorefa import conv2d_Q_fn













device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")

m = nn.Softmax()

cfg = {
    'root': [8, 'M', 16, 'M',16,'M',32,'M','D'],##16,32,32  ##8,16,16 ternary use
    'root1': [8, 'M', 16, 'M', 16, 'M'],  ##16,32,32 no ternary no Dropout
    '2': [16, 'M', 32, 'M', 'D'],
    '3': [16, 'M', 32, 'M', 32, 'D'],
    '4': [16, 32, 'M', 32, 32, 'M', 'D'],
    '5': [16, 32, 'M', 32, 32, 'M', 64, 'D'],
    '6': [16, 32, 32, 'M', 64, 64, 128, 'M', 'D'],
}

cfg_down = {
    '1': [16, 'M', 'D'],
    '2': [16, 'M', 32, 'M', 'D'],
    'root': ['D', 'M', 32, 32, 'M'],
    '4': [16, 32, 'M', 32, 32, 'M', 'D'],
    '5': [16, 32, 'M', 32, 32, 'M', 64, 'D'],
    '6': [16, 32, 32, 'M', 64, 64, 128, 'M', 'D'],
}

# 将所有的图片重新设置尺寸为64*64
w = 64
h = 64
c = 1

############Ternary###################
def Ternarize(tensor):
    output = torch.zeros(tensor.size())
    delta = Delta(tensor)
    alpha = Alpha(tensor,delta)
    for i in range(tensor.size()[0]):
        for w in tensor[i].view(1,-1):
            pos_one = (w > delta[i]).type(torch.FloatTensor)
            neg_one = -1 * (w < -delta[i]).type(torch.FloatTensor)
        out = torch.add(pos_one,neg_one).view(tensor.size()[1:]).to(device)
        #print(output[i].is_cuda)

        output[i] = torch.add(output[i].to(device),torch.mul(out,alpha[i]))
    return output

def Alpha(tensor, delta):
        Alpha = []
        for i in range(tensor.size()[0]):
            count = 0
            abssum = 0
            absvalue = tensor[i].view(1, -1).abs()
            for w in absvalue:
                truth_value = w > delta[i]  # print to see
            count = truth_value.sum()
            #print(truth_value.type(torch.FloatTensor).view(-1, 1).is_cuda)

            abssum = torch.matmul(absvalue, truth_value.type(torch.FloatTensor).view(-1, 1).to(device))
            Alpha.append(abssum / count)
        alpha = Alpha[0]
        for i in range(len(Alpha) - 1):
            alpha = torch.cat((alpha, Alpha[i + 1]))
        return alpha


def Delta(tensor):
    n = tensor[0].nelement()
    if (len(tensor.size()) == 4):  # convolution layer
        delta = 0.7 * tensor.norm(1, 3).sum(2).sum(1).div(n)
    elif (len(tensor.size()) == 2):  # fc layer
        delta = 0.7 * tensor.norm(1, 1).div(n)
    return delta


class TernaryConv2d(nn.Conv2d):
    def __init__(self,*args,**kwargs):
        super(TernaryConv2d,self).__init__(*args,**kwargs)
    def forward(self,input):
        self.weight.data = Ternarize(self.weight.data).to(device)
        out = F.conv2d(input, self.weight, self.bias, self.stride,self.padding, self.dilation, self.groups)
        return out


##########   BNN ##################

class BinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=False,
                 activation_binarizer=ActivationBinarizer(), weight_binarizer=WeightBinarizer()):
        super(BinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.binarize_act = activation_binarizer
        self.binarize_w = weight_binarizer

    def forward(self, input):
        if input.size(1) != 3:
            input = self.binarize_act(input)
        if not hasattr(self.weight, 'original'):
            self.weight.original = self.weight.data.clone()
        self.weight.data = self.binarize_w(self.weight.original)
        #self.weight.data = binarize(self.weight.data)
        out = F.conv2d(input, self.weight, None, self.stride,
                        self.padding, self.dilation, self.groups)

        if self.bias is not None:
            # self.bias.original = self.bias.data.clone() # do we need to save bias copy if it's not quantized?
            out += self.bias.view(1, -1, 1, 1).expand_as(out)
        return out


class InferenceBinaryConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):
        super(InferenceBinaryConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.binarize_act = Sign()

    def forward(self, input):
        if input.size(1) != 3:
            input = self.binarize_act(input)
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)





def binary_conv3x3(in_planes, out_planes, stride=1, groups=1, freeze=False, **kwargs):
    """3x3 convolution with padding"""
    if not freeze:
        return BinaryConv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                            padding=1, bias=False, **kwargs)
    else:
        return InferenceBinaryConv2d(in_planes, out_planes, kernel_size=3, stride=stride, groups=groups,
                                     padding=1, bias=False)












# 读取图片及其标签函数
def read_image(inNum):
    img_data_list = []
    labels = []
    n = 58
    count = 0
    if (inNum == 1):
        path = open(r'Your path/train.txt')
    else:
        path = open(r'Your path/test.txt')

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
        #self.quant = QuantStub()

        #self.dequant = DeQuantStub()
        self.classifier = nn.Sequential(
            nn.Linear(16 * 8 * 8, 4 ,bias=False)##4 classification
        )

    def forward(self, x):##define data pass then return
        # print("origin x: {}".format(x.shape))
        #x = self.quant(x)
        y = self.features(x)
        #y= self.dequant(y)
        # y = self.features_down(x)
        x = y.view(y.size(0), -1)##define 1 dimension
        #print("x is here", x.shape)

        out = self.classifier(x)##then flatten
        #x= self.dequant(out)
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
                #Conv2d = conv2d_Q_fn(w_bit=1)
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding='same',bias=False)]##nn.Conv2d nn.quantized.Conv2d TernaryConv2d
                #layers += [binary_conv3x3(in_channels, x)]##nn.Conv2d nn.quantized.Conv2d binary_conv3x3


                layers += [nn.BatchNorm2d(x),nn.ReLU(inplace=True)]  ##initial in_channels=3


                print("channel_x", x)
                in_channels = x
        #layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        #layers += torch.quantization.DeQuantStub()
        return nn.Sequential(*layers)


def model_one():
    return model('root1')


def train(model, optimizer, loss_fn, trainloader, valloader, device):
    max = 0.20
    for epoch in range(30):##30,40
        model.train()
        count = 0
        total = 0
        for batch_num, (data, target) in enumerate(trainloader):  ##read data==data ##target==label

            # print("batch",batch_num)
            # print("data",data)
            # print("target",target)

            data = data.to(device)  ##use gpu work
            target = target.to(device)  ##use gpu work
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

            optimizer.zero_grad()
            _, net_out = model(data)##net_out == tensor


            loss = loss_fn(net_out, target1)#input,target1
            loss.backward()
            #for p in list(model.parameters()):
             #   if hasattr(p,'original'):
              #      p.data.copy_(p.original)
            optimizer.step()
            #for p in list(model.parameters()):
                #print(hasattr(p,'original'))
              #  if hasattr(p,'original'):
             #       p.original.copy_(p.data.clamp_(-1,1))


            pred = net_out.max(1, keepdim=True)[1]
            #print("pred here",pred)
            #print("target1 here",target1)

            count += pred.eq(target1.view_as(pred)).sum().item()
            #print("count",count)
            total += target.size(0)
            progress_bar(batch_num, len(trainloader), 'Loss: %.4f | Acc: %.3f%% (%d/%d)'
                         % (loss / (batch_num + 1), 100. * count / total, count, total))
        #print("train_loader", len(trainloader.sampler)) all train data
        print("Epoch: ", epoch + 1, "\nTrain: ", count / len(trainloader.sampler))

        model.eval()
        count1 = 0
        for data, target in valloader:
            data = data.to(device)
            target = target.to(device)
            target1 = target.clone().to(device)
            #print("before target1",target1)
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
            #print("after target1",target1)

            _, net_out = model(data)
            pred = net_out.max(1, keepdim=True)[1]
            count1 += pred.eq(target1.view_as(pred)).sum().item()

        print("Val: ", count1 / len(valloader.sampler))
        if count1 / len(valloader.sampler) > max:
            print("checkpoint saved")
            max = count1 / len(valloader.sampler)
            torch.save(model.state_dict(), "CWRU_root.pth")


def average_softmax(model, trainloader, valloader, device):
    nb_classes = 10
    out_classes = 10
    counts = [0 for i in range(nb_classes)]
    soft_out = torch.zeros((nb_classes, 1, nb_classes)).to('cpu')
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
                path = '/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/CNC_Data_Image_orgin_9bit_label/'###CNC_Data_Image_orgin_9bit_label
                path_new = path + line.split('/')[-2] + '/' + line.split('/')[-1].split('.')[-2] + '.tif'
                #print("path_new here",path_new)
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
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params








if __name__ == "__main__":

    train_dataset = MyDataset(txt='/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/train.txt')

    val_dataset = MyDataset(txt='/home/shun/Desktop/Research/MNN-Tree/CWRU/CWRU2D/test.txt')

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               sampler=None,   ##no shuffle
                                               batch_size=64)  ##train_set have something problem

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             sampler=None,   ##no shuffle
                                             batch_size=64)  ##test_set

    train_data, train_label = read_image(1)  # 輸出FE train_data和train_label
    print("train_label", train_label)

    model = model_one().to(device)##root
    model.load_state_dict(torch.load('./CWRU_root.pth'))


    #model_int8 = torch.quantization.quantize_dynamic(model,qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)
    # model.load_state_dict(torch.load('./model_saved/cwru_root.pth'))



    print(summary(model, (1, 64, 64)))
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    #print("stop here now")
    count_parameters(model)
    train(model, optimizer, loss_fn, train_loader, val_loader, device)
    #model_int8 = torch.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None,inplace=False)
    #print("model_here",model.features[0].weight)


    for i in range(3):
        model.load_state_dict(torch.load('./CWRU_root.pth'))

        learning_rate /= 10
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
        loss_fn = nn.CrossEntropyLoss()
        train(model, optimizer, loss_fn, train_loader, val_loader, device)


    #model_int8 = torch.quantization.quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None,inplace=False)
        #print("model", model.features[0].weight)

    model.load_state_dict(torch.load('./CWRU_root.pth'))
    average_softmax(model, train_loader, val_loader, device)

