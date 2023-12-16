import io
import os

import cv2
import numpy as np
from datetime import datetime
from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
from matplotlib import pyplot as plt
from sentinelhub import CRS, BBox, DataCollection, SHConfig, WmsRequest

import model
import logging
import torch
import torch.nn as nn

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(name)s : %(message)s')

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

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


dmodel = model.UNet_Model()


@app.route("/")
def index():
    logging.debug("Index route accessed")
    return "API is Up and Running."


@app.route('/download_image', methods=["GET"])
def download_image():
    config = SHConfig()
    config.instance_id = "93aef06f-6ecd-4b93-b55c-c5492d8a095f"

    # Retrieve latitude and longitude from query parameters
    latitude = float(request.args.get('latitude', 41.9028))

    longitude = float(request.args.get('longitude', 12.4964))

    # Create bounding box around the specified coordinates
    # You might want to adjust the size of the bounding box depending on your requirements
    bbox_lat_offset = 0.023  # size, adjust as needed
    bbox_long_offset = 0.047  # size, adjust as needed
    bbox_coords_wgs84 = (longitude - bbox_long_offset,
                         latitude - bbox_lat_offset,
                         longitude + bbox_long_offset,
                         latitude + bbox_lat_offset)
    # bbox_coords_wgs84 = (12.44693, 41.870072,  12.541001,  41.917096) # Rome

    bbox = BBox(bbox=bbox_coords_wgs84, crs=CRS.WGS84)

    # Retrieve Sentinel-2 data for the specified bounding box
    wms_true_color_request = WmsRequest(
        data_collection=DataCollection.SENTINEL2_L1C,
        layer='TRUE-COLOR-S2L2A',
        bbox=bbox,
        time='latest',
        width=780,
        height=512,
        maxcc=0.2,
        config=config)

    wms_true_color_img = wms_true_color_request.get_data()

    # Take the last image from the list
    # np.save('array_file.npy', image)
    image = wms_true_color_img[-1][:,:,:3][..., ::-1]
    # image_rgb = image_temp[:, :, :3]
    # image = image_rgb[..., ::-1]
    logging.info(type(image), image.shape)
    # Get dimensions of the image
    height, width = image.shape[:2]

    # Calculate the right part of the image to crop to 512x512
    start_row, start_col = int((height - 512) / 2), int(width - 512)

    # Crop the image to 512x512 from the right part
    cropped_image = image[start_row:start_row + 512, start_col:start_col + 512]

    # Downscale the cropped image to 256x256
    resized_image = cv2.resize(cropped_image, (256, 256), interpolation=cv2.INTER_AREA)
    logging.info(resized_image.shape)

    # Convert the result back to RGB
    output = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    # Use a relative path in Replit
    save_directory = 'images'

    # Make sure the directory exists
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    timestamp = datetime.now()

    # Convert the datetime to a timestamp (int)
    timestamp_int = int(timestamp.timestamp())

    filename = f"satellite_image_{timestamp_int}.jpeg"

    # Full path for the image
    file_path = os.path.join(save_directory, filename)

    # Save the processed image
    plt.imsave(file_path, output, format='jpeg')

    dmodel.predict_data(file_path, timestamp_int)

    print(f"Image saved to {file_path}")
    url = f"http://localhost:5000/show_image/{timestamp_int}"

    # Return the Image url to view
    # return f"https://chatgptplugin-vejrekshitij1.replit.app/show_image/{timestamp_int}"
    return jsonify({"url": url})


@app.route('/show_image/<int:timestamp>')
def show_image(timestamp):
    # Get the list of files in the images folder
    image_folder = 'images'  # Change this to the path of your images folder
    image_files = os.listdir(image_folder)
    # Iterate through the files and check if any match the specified timestamp
    for filename in image_files:
        # Check if the filename matches the format "satellite_image_timestamp.png"
        if filename.startswith('output_') and filename.endswith(
                f'{timestamp}.png'):
            # Save the file path
            file_path = f'{image_folder}/{filename}'

            # Send the file as a response with appropriate headers
            return send_file(file_path, mimetype='image/png')

    # If no matching image is found, return an error response
    return jsonify({'error': 'Image not found for the specified timestamp'}), 404


@app.route('/show_folder_images')
def show_folder_images():
    image_folder = 'images'  # Change this to the path of your images folder
    image_files = os.listdir(image_folder)

    # Return a JSON object containing all filenames
    return {"filenames": image_files}


# Provide API for OpenAI with Json data to provide chatbot with description
@app.route('/.well-known/ai-plugin.json')
def serve_ai_plugin():
    return send_from_directory('.',
                               'ai-plugin.json',
                               mimetype='application/json')


# Provide API for OpenAI with yaml configuration file
@app.route('/openapi.yaml')
def serve_openapi_yaml():
    return send_from_directory('.', 'openapi.yaml', mimetype='text/yaml')


if __name__ == '__main__':
    app.run(debug=True, port="5000")
