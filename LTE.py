import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils import MeanShift
from utils import EnhancedHighPass


class LTE(torch.nn.Module):
    def __init__(self, requires_grad=True, rgb_range=1):
        super(LTE, self).__init__()
        ### use vgg19 weights to initialize
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.slice1.parameters():
                param.requires_grad = requires_grad
            for param in self.slice2.parameters():
                param.requires_grad = requires_grad
            for param in self.slice3.parameters():
                param.requires_grad = requires_grad

        vgg_mean = (0.485, 0.456, 0.406)
        vgg_std = (0.229 * rgb_range, 0.224 * rgb_range, 0.225 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

        self.hfm1 = EnhancedHighPass(channels=64)
        self.hfm2 = EnhancedHighPass(channels=128)
        self.hfm3 = EnhancedHighPass(channels=256)

    def forward(self, x):
        # print("input shape ",x.shape)
        x = self.sub_mean(x)
        x = self.slice1(x)
        x_lv1 = x
        x_lv1 = self.hfm1(x_lv1)

        # print("after slice1:",x_lv1.shape)
        x = self.slice2(x)
        x_lv2 = x
        x_lv2 = self.hfm2(x_lv2)
        #  print("after slice1:", x_lv2.shape)
        x = self.slice3(x)
        x_lv3 = x
        x_lv3 = self.hfm3(x_lv3)

        # print("after slice1:", x_lv3.shape)

        # Apply HFM to enhanced features
        #
        # print("after HFM1:", x_lv1.shape)
        #
        # print("after HFM2:", x_lv2.shape)
        #
        # print("after HFM3:", x_lv3.shape)

        return x_lv1, x_lv2, x_lv3
