

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

    def __init__(self, dataPath, trsf=None):


        self.len = 0

    def __len__(self):
        return self.len

    def __getitem__(self, idx):

        return None


##### Dataloader
def getDataloader(args, val=False):

    trsf = None
    
    faceData = FaseDataset(args.dataPath, trsf)
    dataloader = DataLoader(faceData, batch_size=args.batchSize, shuffle=args.train, num_workers=args.numWorkers)

    return dataloader


