import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Function
from .binarized_modules import  BinarizeLinear_train,BinarizeConv2d_train,BinarizeConv2d,BinarizeLinear



class VGG_Cifar10(nn.Module):

    def __init__(self, num_classes=10,w_bit=3,a_bit=1,comp=False):
        super(VGG_Cifar10, self).__init__()
        self.infl_ratio=1;
        self.lr = 2.5e-2; 
        self.w_bit = w_bit
        self.a_bit = a_bit
        self.comp = comp


        self.features = nn.Sequential(
            BinarizeConv2d_train(3, 128*self.infl_ratio, kernel_size=3, stride=1, padding=1, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),

            BinarizeConv2d_train(128*self.infl_ratio, 128*self.infl_ratio, kernel_size=3, padding=1, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(128*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d_train(128*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d_train(256*self.infl_ratio, 256*self.infl_ratio, kernel_size=3, padding=1, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(256*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d_train(256*self.infl_ratio, 512*self.infl_ratio, kernel_size=3, padding=1, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.BatchNorm2d(512*self.infl_ratio),
            nn.Hardtanh(inplace=True),


            BinarizeConv2d_train(512*self.infl_ratio, 512, kernel_size=3, padding=1, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(512),
            nn.Hardtanh(inplace=True)

        )
        self.classifier = nn.Sequential(
            BinarizeLinear_train(512 * 4 * 4, 1024, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear_train(1024, 1024, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.BatchNorm1d(1024),
            nn.Hardtanh(inplace=True),
            #nn.Dropout(0.5),
            BinarizeLinear_train(1024, num_classes, bias=False, precision_w = self.w_bit, precision_a = self.a_bit, comp =self.comp),
            nn.BatchNorm1d(num_classes, affine=False),
            nn.LogSoftmax()
        )

        self.regime = {
            0: {'optimizer': 'Adam', 'betas': (0.9, 0.999),'lr': self.lr},
            40: {'lr': self.lr/5},
            80: {'lr': self.lr/10},
            100: {'lr': self.lr/50},
            120: {'lr': self.lr/100},
            140: {'lr': self.lr/500}
        }
        
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 512 * 4 * 4)
        x = self.classifier(x)
        return x


def vgg_cifar10_train(**kwargs):
    num_classes, depth, dataset, w_bit, a_bit, comp = map(
        kwargs.get, ['num_classes', 'depth', 'dataset', 'w_bit', 'a_bit', 'comp'])

    num_classes = kwargs.get( 'num_classes', 10)
    return VGG_Cifar10(num_classes,w_bit,a_bit,comp)
