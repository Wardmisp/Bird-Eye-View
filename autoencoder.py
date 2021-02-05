# Import the different module we will need in this notebook
import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import random
from math import *

def padding_same(output_dim,input_dim, kernel, stride):
  return ceil((output_dim-((input_dim-kernel)/stride+1)*stride/2))


class get_model(nn.Module):
    
    def __init__(self, classes=10):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(7, 32, kernel_size=5, padding=[padding_same(48,96,5,2),padding_same(72,144,5,2)], stride = 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=[padding_same(24,48,5,2),padding_same(36,72,5,2)], stride = 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=[padding_same(12,24,3,2),padding_same(18,36,3,2)], stride = 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=[padding_same(6,12,3,2),padding_same(9,18,3,2)], stride = 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=(1,1), stride = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=(1,1), stride = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
        self.layer7 = nn.Sequential(
            nn.Upsample(scale_factor  = (2,2)),
            nn.Conv2d(256, 256, kernel_size=3, padding=(1,1), stride = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
        self.layer8 = nn.Sequential(
            nn.Upsample(scale_factor  = (2,2)),
            nn.Conv2d(256, 128, kernel_size=3, padding=(1,1), stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
        self.layer9 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=(1,1), stride = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
        self.layer10 = nn.Sequential(
            nn.Upsample(scale_factor  = (2,2)),
            nn.Conv2d(128, 64, kernel_size=3, padding=(1,1), stride = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
        self.layer11 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=(1,1), stride = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
        self.layer12 = nn.Sequential(
            nn.Upsample(scale_factor = (2,1)),
            nn.Conv2d(32, 32, kernel_size=5, padding=(2,2), stride = 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2,inplace=True))
        #last layer
        self.layer13 = nn.Sequential(
            nn.Conv2d(32, 7, kernel_size=5, padding=(2,2), stride = 1),
            nn.Softmax()
        )
        
    def forward(self, x):
        
      out = self.layer1(x)
      out = self.layer2(out)
      out = self.layer3(out)
      out = self.layer4(out)
      out = self.layer5(out)
      out = self.layer6(out)
      out = self.layer7(out)
      out = self.layer8(out)
      out = self.layer9(out)
      out = self.layer10(out)
      out = self.layer11(out)
      out = self.layer12(out)
      out = self.layer13(out)
      return out