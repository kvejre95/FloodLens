<div align="center">

  # FloodLense: ChatGPT Plugin for Flood Detection

![](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white)
![](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![](https://img.shields.io/badge/react-%2320232a.svg?style=for-the-badge&logo=react&logoColor=%2361DAFB)

</div>

FloodLense is an innovative ChatGPT plugin designed to detect flood-affected areas using satellite imagery. At the heart of FloodLense is a sophisticated UNet model, meticulously trained to identify and analyze flood regions in satellite images. This model has been expertly integrated into a FastAPI backend, creating a seamless bridge between the Sentinel Hub API and our advanced machine learning capabilities.

The workflow begins with the FastAPI making API calls to Sentinel Hub, retrieving the latest satellite imagery based on specific geographic coordinates. These images are then processed and fed into the trained UNet model, which meticulously analyzes them to pinpoint flood-affected areas. The results from this analysis provide invaluable insights, offering a near real-time assessment of flood impact.

Designed as a ChatGPT plugin, FloodLense enhances the capabilities of ChatGPT, allowing it to respond to queries related to flood detection and assessment with precise, AI-driven data directly sourced from up-to-date satellite imagery. This integration makes FloodLense not just a tool for analysis, but also a means for interactive, AI-powered communication, offering users immediate access to critical flood-related information.

FloodLense is more than just a technical achievement; it's a step forward in disaster management and response, harnessing the power of AI and satellite technology to aid in timely and effective decision-making.

## Prerequisites
Before running this project, ensure you have the following accounts and tools set up:

- **Sentinel Hub Account**: Create a new Configuration Template and save the Instance ID for use in the backend code. [Create Account](https://www.sentinel-hub.com/)
- **OpenAI Account**: Register for an OpenAI account, add funds, and obtain your API Key. [Sign Up](https://openai.com/)
- **Training Dataset**: The dataset required for training the model is available at the following link: 1. [Kaggle Data](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies) and 2. [FloodNet Data](https://github.com/BinaLab/FloodNet-Supervised_v1.0) 

## Installation and Setup

### Step 1: Preprocess the Data
- Navigate to the `ML Code/utils` folder.
- Run the preprocessing scripts on the dataset.
- Dataset for preprocessing is available at the above provided links.

### Step 2: Train the UNet Model
- Using the preprocessed data, train the UNet model.
- Save the trained model as `UNet_ckpt_1.pt`.
- Move the saved model file to the `fastapi-backend` folder.

### Step 3: Run the FastAPI Backend
- Change directory to the `fastapi-backend` folder.
- Install all required Python packages (a `requirements.txt` file is recommended).
- Execute the FastAPI server code.

### Step 4: Run OpenAI Script
- In the `OpenAI-Script` folder, execute the script for interacting with the OpenAI API.

### Step 5: Start the React Frontend Application
- For a better user interface, navigate to the `React-Frontend` folder.
- Run the React application.

## Project Structure

- `ML Code/utils`: Contains utility scripts for data preprocessing.
- `fastapi-backend`: Includes the FastAPI server code and the trained UNet model.
- `OpenAI-Script`: Contains scripts for OpenAI API interaction.
- `React-Frontend`: The front-end React application for a user-friendly interface.


