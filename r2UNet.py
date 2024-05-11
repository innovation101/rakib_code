import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
class UpConv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=True),
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class U_Net3D(nn.Module):
    def __init__(self, img_ch=3, output_ch=2):
        super(U_Net3D, self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.Conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.Conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.Conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.Conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.Up5 = UpConv(ch_in=1024, ch_out=512)
        self.Up_conv5 = ConvBlock(ch_in=1024, ch_out=512)

        self.Up4 = UpConv(ch_in=512, ch_out=256)
        self.Up_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.Up3 = UpConv(ch_in=256, ch_out=128)
        self.Up_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.Up2 = UpConv(ch_in=128, ch_out=64)
        self.Up_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.Final_conv = nn.Conv3d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        output = self.Final_conv(d2)
        return output