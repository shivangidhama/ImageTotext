# Handwritten Text Recognition App

## Overview

This is a Streamlit app that uses the TrOCR model to recognize and transcribe handwritten text from images. The app allows users to upload images containing handwritten text and get the transcribed text as output.

## Features

- **Image Upload**: Upload an image containing handwritten text.
- **Text Recognition**: The app uses the TrOCR model to transcribe the text from the uploaded image.
- **Display Results**: View the recognized text directly within the app.

## Requirements

To run this app locally, you need to have the following Python packages installed:

- `streamlit`
- `transformers`
- `pillow`
- `torch`

You can install these packages using the `requirements.txt` file provided:

```bash
pip install -r requirements.txt

