'''
Code to test the output shape of the UNet segmentation model
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, std_init=None, dropout=0.2, batch_norm=True):
        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=(not batch_norm)),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=(not batch_norm)),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_val=0.2, std_init=None):
        super(UNet, self).__init__()

        filters = 32

        self.enc1 = DoubleConv(in_channels, filters, std_init)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = DoubleConv(filters, filters*2, std_init)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(filters*2, filters*4, std_init)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(filters*4, filters*8, std_init)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = DoubleConv(filters*8, filters*16, std_init)
        self.pool5 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(filters*16, filters*32, std_init)

        self.up5 = nn.ConvTranspose2d(filters*32, filters*16, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(filters*32, filters*16, std_init, dropout_val)

        self.up4 = nn.ConvTranspose2d(filters*16, filters*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(filters*16, filters*8, std_init, dropout_val)

        self.up3 = nn.ConvTranspose2d(filters*8, filters*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(filters*8, filters*4, std_init, dropout_val)

        self.up2 = nn.ConvTranspose2d(filters*4, filters*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(filters*4, filters*2, std_init, dropout_val)

        self.up1 = nn.ConvTranspose2d(filters*2, filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(filters*2, filters, std_init, dropout_val)

        self.final = nn.Conv2d(filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        enc5 = self.enc5(self.pool4(enc4))

        bottleneck = self.bottleneck(self.pool5(enc5))

        dec5 = self.dec5(torch.cat((self.up5(bottleneck), enc5), 1))
        dec4 = self.dec4(torch.cat((self.up4(dec5), enc4), 1))
        dec3 = self.dec3(torch.cat((self.up3(dec4), enc3), 1))
        dec2 = self.dec2(torch.cat((self.up2(dec3), enc2), 1))
        dec1 = self.dec1(torch.cat((self.up1(dec2), enc1), 1))

        return self.sigmoid(self.final(dec1))

# Usage
print('started')

# Setup dummy inputs
batch_size = 1
channels = 1
height = 256
width = 256

x = torch.randn(batch_size, channels, height, width)
print('created dummy input')

device = "mps"
device = torch.device("mps")

print('sending to device')
# Send to device
x = x.to(device)

# Forward pass
model = UNet(in_channels=1, out_channels=1).to(device)
outputs = model(x)

# Check outputs
print(outputs.shape)



 