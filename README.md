# Object Classifier and Interactive Experience

This is a Streamlit application that uses a pre-trained deep learning model to classify objects in images. It provides an interactive experience where users can take a picture, get predictions, fetch related YouTube videos, and answer questions based on the predicted object.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Running the App](#running-the-app)
- [Application Flow](#application-flow)
- [Folder Structure](#folder-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

Make sure you have the following installed:

- [Python](https://www.python.org/downloads/)
- [Streamlit](https://docs.streamlit.io/installation)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Running the App
Execute the following command to run the Streamlit app:

```bash
streamlit run app.py
```
This will launch the app in your default web browser. Follow the instructions on the webpage to interact with the application.

Application Flow
Take a Picture: Click the "Take a picture" button to capture an image using your device's camera.

Object Prediction: The app will use a pre-trained model to predict the class of the object in the image.

YouTube Video Fetching: Click "Fetch Youtube Video" to find related YouTube videos based on the predicted object.

Answer Questions: Click "Fetch Questions" to answer a set of predefined questions related to the predicted object.

Folder Structure
dataset/: Contains training and testing datasets.
models/: Store your pre-trained model file here.
app.py: Main application script.
requirements.txt: List of Python dependencies.
Customization
Model: You can replace the pre-trained model file (object_classifier.h5) in the models/ directory with your own model.

Dataset: Customize the train_data_dir and valid_data_dir paths in app.py according to your dataset structure.

Questions: Modify the sets of questions (qs1 and qs2) in app.py to suit your needs.

Contributing
Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or create a pull request.

License
This project is licensed under the MIT License.
