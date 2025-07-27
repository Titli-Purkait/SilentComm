# SilentComm 🔇💬

**SilentComm** is a real-time facial emotion recognition system designed to assist in silent or non-verbal communication using AI and computer vision. Built with Python, TensorFlow, and Streamlit, it uses webcam input to detect and classify facial expressions.

This project is built using a Convolutional Neural Network (CNN) trained on the FER-2013 dataset and inspired by the Kaggle notebook:  
🔗 [Facial Emotion Recognition by gauravece068](https://www.kaggle.com/code/gauravece068/facial-emotion-recognition)

## 🚀 Features

- 🎥 Real-time webcam input
- 😃 Facial expression recognition using a trained CNN model
- 📊 Displays predicted emotion and confidence
- 🧠 Deep learning model saved in `.h5` format
- 🧪 Integrated with Streamlit for easy UI

## 📦 Tech Stack

- **Frontend**: HTML, JavaScript (for webcam capture)
- **Backend**: Python, Flask / Streamlit
- **AI Model**: TensorFlow/Keras
- **Others**: OpenCV, NumPy, Pillow, h5py, Streamlit-WebRTC

## 📁 Project Structure

SilentComm/
│
├── model.h5 # Trained CNN model
├── streamlit_app.py # Streamlit UI and video processing logic
├── templates/ # HTML templates if Flask used
├── static/ # JS and CSS files
├── dataset/ # Training images
├── utils/ # Preprocessing or helper scripts
└── README.md # Project documentation


## 🧠 How the Model Works

The emotion recognition model is a Convolutional Neural Network (CNN) trained on labeled facial expression images. It classifies emotions into categories like:

- Happy 🙂
- Sad 😢
- Angry 😠
- Neutral 😐
- Surprised 😮

by Team ITRONIX
