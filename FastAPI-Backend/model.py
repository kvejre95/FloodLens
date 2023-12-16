import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import cv2
from PIL import Image


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

        self.enc2 = DoubleConv(filters, filters * 2, std_init)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = DoubleConv(filters * 2, filters * 4, std_init)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = DoubleConv(filters * 4, filters * 8, std_init)
        self.pool4 = nn.MaxPool2d(2)

        self.enc5 = DoubleConv(filters * 8, filters * 16, std_init)
        self.pool5 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(filters * 16, filters * 32, std_init)

        self.up5 = nn.ConvTranspose2d(filters * 32, filters * 16, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(filters * 32, filters * 16, std_init, dropout_val)

        self.up4 = nn.ConvTranspose2d(filters * 16, filters * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(filters * 16, filters * 8, std_init, dropout_val)

        self.up3 = nn.ConvTranspose2d(filters * 8, filters * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(filters * 8, filters * 4, std_init, dropout_val)

        self.up2 = nn.ConvTranspose2d(filters * 4, filters * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(filters * 4, filters * 2, std_init, dropout_val)

        self.up1 = nn.ConvTranspose2d(filters * 2, filters, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(filters * 2, filters, std_init, dropout_val)

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


class UNet_Model():

    def __init__(self):
        self.target_size = (256, 256)
        self.model = UNet(in_channels=3, out_channels=1)
        self.model = torch.load('UNet_ckpt_1.pt', map_location=torch.device('cpu'))
        # self.model.eval()

    def process_image(self, image):
        """
        Process the image by converting it from PyTorch format to standard image format.
        Also, resize the image if necessary.
        """
        print(image.shape)
        # Convert from PyTorch format (C, H, W) to standard format (H, W, C)
        image = np.moveaxis(image.numpy(), 0, -1)
        print(image.shape)

        # Resize the image if it's not already 256x256
        if image.shape[0] != 256 or image.shape[1] != 256:
            image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

        return image

    def resize_mask(self, mask, size):
        # Convert the mask to grayscale
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        # Resize the mask to the specified size
        resized_mask = cv2.resize(mask_gray, size, interpolation=cv2.INTER_NEAREST)
        resized_mask = np.expand_dims(resized_mask, axis=2)
        return resized_mask

    def resize_image(self, image, size):
        # Resize the image to the specified size
        resized_image = cv2.resize(image, size)
        return resized_image

    def preprocess_data(self, image_path):
        logging.info(image_path)
        image = plt.imread(image_path).astype(np.float32) / 255.
        logging.info(image.shape)
        resized_image = self.resize_image(image, self.target_size)
        logging.info(resized_image.shape)
        resized_image = np.moveaxis(resized_image, -1, 0).reshape(1, 3, 256, 256)
        return resized_image

    def predict_data(self, image_path, timestamp):
        data = torch.from_numpy(self.preprocess_data(image_path)).to("cpu")
        pred = self.model(data)
        pred = pred.cpu().detach().numpy().reshape(1, 1, 256, 256)
        pred = pred.reshape(256, 256)
        # pred = pred > 0.9
        # Creating an RGBA version of the original image for transparency
        processed_image = self.process_image(data[0])
        rgba_image = np.zeros((256, 256, 4))
        rgba_image[:, :, :3] = processed_image
        rgba_image[:, :, 3] = 1  # Fully opaque

        # Creating a red transparent overlay
        overlay = np.zeros((256, 256, 4))
        overlay[:, :, 0] = 1  # Red channel
        overlay[:, :, 3] = pred * 0.5  # 50% transparency where pred is True

        # Applying the overlay
        segmented_image_with_transparency = rgba_image.copy()
        for i in range(4):
            segmented_image_with_transparency[:, :, i] = rgba_image[:, :, i] * (1 - overlay[:, :, 3]) + overlay[:, :,
                                                                                                        i] * overlay[:,
                                                                                                             :, 3]

        print(segmented_image_with_transparency.shape)
        if segmented_image_with_transparency.dtype == 'float':
            print(segmented_image_with_transparency.dtype)
            image_np = (segmented_image_with_transparency * 255).astype(np.uint8)
        image = Image.fromarray(image_np, 'RGBA')
        image.save(f'images/output_{timestamp}.png')
        # return segmented_image_with_transparency


# dmodel = Diffusion_Model()
# #
# plt.subplot(1, 2, 2)
# plt.imshow(dmodel.predict_data("sample.jpg"))
# plt.title('Segmented Image with Highlight')
# plt.axis('off')
#
# plt.show()
test = "Hello world!  Your web application is working! - Vejre"
