# Crack Detection Model

This project is a binary classification model that detects cracks in structures using image data. The model uses a Convolutional Neural Network (CNN) and is implemented in Keras.

## Project Structure
- `app.py`: The Streamlit app for uploading and testing new images.
- `model/`: Contains all model-related files.
  - `model_training.py`: Model training script.
  - `utils.py`: Helper functions for data processing.
  - `config.py`: Configurations and hyperparameters.
  - `model.h5`: Saved trained model.
- `images/`: Folder to store test images for the app.

## Setup Instructions

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the Streamlit app: `streamlit run app.py`

## Usage
- Train the model by running `model_training.py`.
- Launch `app.py` to test the model on new images.

## Requirements
See `requirements.txt` for the required Python packages.
