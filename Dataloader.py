

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
from PIL import Image

from utils import listAllImg




##### Train Dataset
class FaceDataset(Dataset):

    def __init__(self, dataPath, batchSize):
        self.dataPath = dataPath
        self.batchSize = batchSize
        self.dataList = listAllImg(self.dataPath)
        self.len = len(self.dataList)


        self.trsf32 = transforms.Compose([
                    transforms.Resize([32,32]),
                    transforms.ToTensor(),
                ])
        self.trsf64 = transforms.Compose([
                    transforms.Resize([64,64]),
                    transforms.ToTensor(),
                ])
        self.trsf128 = transforms.Compose([
                    transforms.Resize([128,128]),
                    transforms.ToTensor(),
                ])



    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        dataName = self.dataList[idx]
        img = Image.open(dataName)
        img32 = self.trsf32(img)
        img64 = self.trsf64(img)
        img128 = self.trsf128(img)

        noise = torch.randn(100, 1, 1)


        return noise, img32, img64, img128


##### Dataloader
def getDataloader(args):
    faceData = FaceDataset(args.dataPath, args.batchSize)
    dataloader = DataLoader(faceData, batch_size=args.batchSize, shuffle=args.train, num_workers=args.numWorkers, drop_last=True)

    return dataloader


