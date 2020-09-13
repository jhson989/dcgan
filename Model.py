import torch
import torch.optim as optim
import torch.nn as nn
from Dataloader import getDataloader
from utils import saveImage

from model.dcgan import Generator, Discriminator

class Model:

    def __init__(self, args, logger):

        ## Configuration
        self.args = args
        self.device = "cuda" if args.ngpu > 0 else "cpu"
        self.logger = logger
    
        ## Network 
        self.netG = Generator().to(self.device)
        self.netD = Discriminator().to(self.device)

    def train(self):

        ## Optim
        self.optimG = optim.Adam(self.netG.parameters(), lr=self.args.lr[0], betas=(self.args.beta1, self.args.beta2))
        self.optimD = optim.Adam(self.netD.parameters(), lr=self.args.lr[1], betas=(self.args.beta1, self.args.beta2))  

        self.criterion = nn.BCELoss()    


        ## Data
        self.realLabel = torch.ones((self.args.batchSize)).to(self.device)
        self.fakeLabel = torch.zeros((self.args.batchSize)).to(self.device)
        trainLoader = getDataloader(self.args)


        
        for epoch in range(self.args.numEpoch):
            for i, (latent, face32, face64, face128) in enumerate(trainLoader):
                latent = latent.to(self.device)
                face32, face64, face128 = face32.to(self.device), face64.to(self.device), face128.to(self.device) 

                fake32, fake64, fake128 = self.netG(latent)

                dLoss = self.trainD(
                        [fake32.detach(), fake64.detach(), fake128.detach()],
                        [face32, face64, face128]
                    )

                gLoss = self.trainG(
                        [fake32, fake64, fake128]
                    )

                if i % 1 == 0:
                    self.logger.log("[%3d/%3d]][%5d/%5d] : D32(%.3f = F%.3f + R%.3f) D64(%.3f = F%.3f + R%.3f)  D128(%.3f = F%.3f + R%.3f) G(%.3f, %.3f, %.3f)" 
                        % (epoch, self.args.numEpoch, i, len(trainLoader), 
                            dLoss[0]+dLoss[3], dLoss[0], dLoss[3],
                            dLoss[1]+dLoss[4], dLoss[1], dLoss[4],
                            dLoss[2]+dLoss[5], dLoss[2], dLoss[5],
                            gLoss[0], gLoss[1], gLoss[2]
                           )
                        )

                saveImage(self.args, epoch, i, [face32, face64, face128, fake32, fake64, fake128], 20)

            if epoch % 10 == 0:
                torch.save(self.netG.state_dict(), self.args.savePath+"G_%d.pth"%epoch)
                torch.save(self.netD.state_dict(), self.args.savePath+"D_%d.pth"%epoch)


    def trainD(self, fakes, faces):
        self.netD.zero_grad()

        # Fake
        dLoss = []
        for fake in fakes:
            output = torch.squeeze(self.netD(fake))
            fakeLoss = self.criterion(output, self.fakeLabel)
            fakeLoss.backward()
            dLoss.append(fakeLoss.item())

        # Real
        for face in faces:
            output = torch.squeeze(self.netD(face))
            realLoss = self.criterion(output, self.realLabel)
            realLoss.backward()
            dLoss.append(realLoss.item())

        self.optimD.step()
        return dLoss



    def trainG(self, fakes):
        self.netG.zero_grad()

        gLoss = []
        for fake in fakes:
            output = torch.squeeze(self.netD(fake))
            loss = self.criterion(output, self.realLabel)
            loss.backward(retain_graph=True)
            gLoss.append(loss.item())

        self.optimG.step()
        return gLoss

