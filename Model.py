import torch
import torch.optim as optim
import torch.nn as nn
from Dataloader import getDataloader
from utils import saveImage


class Model:

    def __init__(self, args, logger):

        ## Configuration
        self.args = args
        self.device = "cuda" if args.ngpu > 0 else "cpu"
        self.logger = logger
    
        ## Network 
    def train(self):

        print("train")
