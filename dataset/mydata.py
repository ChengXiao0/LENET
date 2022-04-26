import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2 as cv
import numpy as np
import os


class mydata(Dataset):
    def __init__(self, path):
        self.imag1set = np.array([x.name for x in os.scandir(path)])
        self.path = path
        # str = "/home/xiaoxiao/PycharmProjects/imgResize/dataset/data/" + self.imag1set[1]
        # x = cv.imread(str)
        # x = cv.resize(x, (32, 32))
        # x = cv.cvtColor(x, cv.COLOR_RGB2GRAY)
        # x = torch.tensor(x)
        # print(x.size())
        # x = torch.squeeze(x)
        # print(x.size())
        # x = torch.unsqueeze(x, 0)
        # print(x.size())



    def __getitem__(self, item):
        str = self.path + self.imag1set[item]
        label = eval(self.imag1set[item][0])
        x = cv.imread(str)
        x = cv.resize(x, (64, 64))
        x = cv.cvtColor(x, cv.COLOR_RGB2GRAY)
        x = torch.tensor(x)
        x = torch.unsqueeze(x, 0)
        # x = transforms.ToTensor(x)
        return label, x

    def __len__(self):
        return len(self.imag1set)


