
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResSeq(nn.Module):

    def __init__(self, inC, outC, out="ReLU"):
        super(ResSeq, self).__init__()
        
        self.seq = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1),
            nn.BatchNorm2d(outC),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1),
        )

        self.res = nn.Conv2d(inC, outC, 1)
        if out == "ReLU":
            self.out = nn.LeakyReLU(0.2, True)
        elif out == "Sigmoid":
            self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.seq(x) + self.res(x)
        return self.out(x)




class ConvSeq(nn.Module):

    def __init__(self, inC, outC):
        super(ConvSeq, self).__init__()
        
        self.seqConv = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1),
            nn.BatchNorm2d(outC),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1),
            nn.BatchNorm2d(outC),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.seqConv(x)


        
#############################################################################################################3
############################################### Generator ###################################################3
#############################################################################################################3

class Out(nn.Module):

    def __init__(self, inC, outC):
        super(Out, self).__init__()

        '''
        self.outSeq = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1),
            nn.BatchNorm2d(outC),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        '''
        self.outSeq1 = ResSeq(inC, inC)
        self.outSeq2 = ResSeq(inC, outC, "Sigmoid")

    def forward(self, x):
        x = self.outSeq1(x)
        x = self.outSeq2(x)
        return x


class Up(nn.Module):

    def __init__(self, inC, outC, s=2, p=1):

        super(Up, self).__init__()
        self.up = nn.Sequential(
                nn.ConvTranspose2d(inC, outC, kernel_size=4, stride=s, padding=p),
#                ConvSeq(outC, outC), 
                ResSeq(outC, outC)
           )

    def forward(self, x):
        x = self.up(x)
        return x


class Generator(nn.Module):

    def __init__(self):

        super(Generator, self).__init__()
        
        nC = 64

        self.up1 = Up(100, nC*8, 1, 0) # 4
        self.up2 = Up(nC*8, nC*8) # 8
        self.up3 = Up(nC*8, nC*4) # 16
        self.up4 = Up(nC*4, nC*2) # 32
        self.up5 = Up(nC*2, nC*2) # 64
        self.up6 = Up(nC*2, nC*2) # 64

        self.out32 = Out(nC*2, 3) # 32
        self.out64 = Out(nC*2, 3) # 64
        self.out128 = Out(nC*2, 3) # 128


    def forward(self, x):

        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x32 = self.up4(x)
        x64 = self.up5(x32)
        x128 = self.up6(x64)

        return self.out32(x32), self.out64(x64), self.out128(x128)




#############################################################################################################3
################################################ Discriminator ##############################################3
#############################################################################################################3

class In(nn.Module):

    def __init__(self, inC, outC):
        super(In, self).__init__()
        
        '''
        self.inSeq = nn.Sequential(
            nn.Conv2d(inC, outC, kernel_size=3, padding=1),
            nn.BatchNorm2d(outC),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(outC, outC, kernel_size=3, padding=1),
        )
        '''
        self.inSeq = ResSeq(inC, outC)
    def forward(self, x):
        return self.inSeq(x)



class Down(nn.Module):

    def __init__(self, inC, outC, s=2, p=1, out="ReLU"):
        super(Down, self).__init__()

        self.down = nn.Sequential(
#                ConvSeq(inC, outC), 
                ResSeq(inC, outC),
                nn.Conv2d(outC, outC, kernel_size=3, stride=s, padding=p),
           )

        if out == "ReLU":
            self.out = nn.LeakyReLU(0.2, True)
        elif out == "Sigmoid":
            self.out = nn.Sigmoid()



    def forward(self, x):
        return self.out(self.down(x))


class Discriminator(nn.Module):

    def __init__(self):

        super(Discriminator, self).__init__()

        nC = 32

        self.down1 = Down(3, nC*1) # 64
        self.down2 = Down(nC*1, nC*2) # 32
        self.down3 = Down(nC*2, nC*4) # 16
        self.down4 = Down(nC*4, nC*8) # 8
        self.down5 = Down(nC*8, nC*16) # 4
        self.down6 = Down(nC*16, 1, 4, 0, "Sigmoid") # 1

        self.in32 = In(3, nC*2)
        self.in64 = In(3, nC*1)


    def forward(self, x):

        if x.size()[2] == 32:
            x = self.in32(x)

        elif x.size()[2] == 64:
            x = self.in64(x)
            x = self.down2(x)
        elif x.size()[2] == 128:
            x = self.down1(x)
            x = self.down2(x)


        x = self.down3(x)
        x = self.down4(x)
        x = self.down5(x)
        x = self.down6(x)

        return x
