import argparse, os
import random

import torch

from Model import Model
from utils import Logger

def argParsing():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=bool, default=True, help="train mode")

    ## Training policy
    parser.add_argument("--numEpoch", type=int, default=2000, help="num of epoch")
    parser.add_argument("--batchSize", type=int, default=2, help="input batch size")
    parser.add_argument("--lr", nargs="+", type=float, default=[(2e-3),(1e-3),(1e-3),(2e-3)], help="learing rate : Gen GANG, GANL, SR")
    parser.add_argument("--beta1", type=float, default=0.9, help="beta1 for adam")
    parser.add_argument("--beta2", type=float, default=0.999, help="beta2 for adam")
    parser.add_argument("--manualSeed", type=int, default=1, help="manual seed")
    ## Environment
    parser.add_argument("--ngpu", type=int, default=1, help="number of gpus")
    parser.add_argument("--numWorkers", type=int, default=4, help="number of workers for dataloader")
    ## Data
    parser.add_argument("--dataPath", type=str, default="./data/", help="path to dataset") 
    parser.add_argument("--savePath", type=str, default="./result/1/", help="path to save folder") 
    args = parser.parse_args()
    return args

def setEnv(args):

    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    try:
        if not os.path.exists(args.savePath):
            os.makedirs(args.savePath)
    except OSError:
        print("Error: Creating save folder. [" + args.savePath + "]")

    if torch.cuda.is_available() == False:
        args.ngpu = 0


if __name__ == "__main__":

    args = argParsing()
    setEnv(args)
    logger = Logger(args.savePath)
    logger.log(str(args))

    model = Model(args, logger)
    if args.train == True:
        logger.log("[[Train]]")
        model.train()
    else:
        logger.log("[[Demo]]")





