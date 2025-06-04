# Aerial Photo Classification Streamlit App

A Streamlit web application for classifying aerial photographs using a pre-trained deep learning model. Users can upload an image, and the app will predict its category (e.g., forest, urban, water, agricultural land).

## Table of Contents
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Setup and Installation](#setup-and-installation)
- [Configuration](#configuration)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

## Features
* Upload aerial images (JPG, JPEG, PNG).
* Preprocesses images including resizing, noise reduction, and histogram equalization.
* Predicts the class of the aerial photo using a pre-trained model.
* Displays the uploaded image and the prediction result with confidence.

## Tech Stack
* Python (3.8+)
* Streamlit
* TensorFlow / Keras
* OpenCV (cv2)
* Pillow (PIL)
* NumPy

## Prerequisites
Before you begin, ensure you have the following installed on your system:
* **Git:** To clone the repository.
* **Python:** Version 3.8 or higher is recommended. You can download it from [python.org](https://www.python.org/).
* **pip:** Python package installer (usually comes with Python).

## Setup and Installation

Follow these steps to set up the project locally:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rubricate12/TUBES_PCD_SKYVIEW.git
    cd TUBES_PCD_SKYVIEW
    ```

2.  **Create and activate a virtual environment (recommended):**
    This keeps your project dependencies isolated.
    ```bash
    # For Windows
    python -m venv venv
    venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    All required Python packages are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the Pre-trained Model:**
    *  The model file = should already be in the project directory. Ensure it's in the correct path referenced by the application (e.g., root folder or a `models/` subfolder).
    *  Place the downloaded model file into the root directory of this project, or update the `MODEL_PATH` variable in `app.py` (or your main script) to point to its location.

## Configuration
Before running the application, you might want to check or adjust the following configurations within your main Streamlit Python script (e.g., `app.py`):

* `MODEL_PATH`: Ensure this variable points to the correct path of your `.h5` model file.
* `IMG_HEIGHT` and `IMG_WIDTH`: These should match the expected input dimensions of your pre-trained model.
* `CLASS_NAMES`: This list should contain the names of the classes your model predicts, in the correct order.

## Running the Application

Once the setup and configuration are complete:

1.  Ensure your virtual environment is activated.
2.  Navigate to the root project directory in your terminal.
3.  Run the Streamlit application using the following command:
    ```bash
    streamlit run app.py
    ```
    *(Replace `app.py` with the actual name of your main Streamlit script if it's different.)*

4.  The application should automatically open in your default web browser. If not, navigate to the local URL displayed in the terminal (usually `http://localhost:8501`).

## Project Structure
A brief overview of the key files and directories:
├── app.py                # Main Streamlit application script
├── your_model.h5         # Pre-trained model file (if included or where to place it)
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── assets/               # (Optional) For images, logos, etc.
└── ...                   # Other project files or notebooks


## Troubleshooting
* **Model Loading Error:**
    * Ensure the `MODEL_PATH` in `app.py` is correct and the `.h5` file is present at that location.
    * Verify that your TensorFlow/Keras version is compatible with the version used to save the model (see previous discussions on TF version mismatches if this occurs).
* **Dependency Issues:**
    * If `pip install -r requirements.txt` fails, ensure your Python and pip are up to date. Try installing problematic packages individually.
* **OpenCV Issues:**
    * Sometimes OpenCV requires additional system libraries. If you encounter issues related to `cv2`, ensure you installed `opencv-python` or `opencv-python-headless`.

---
