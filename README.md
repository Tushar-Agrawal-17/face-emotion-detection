Face Emotion Detection using Deep Learning

A Real-Time Face Emotion Detection System built using Python, OpenCV, and Deep Learning (CNN with TensorFlow/Keras).
The system captures video from a webcam, detects human faces, and predicts emotions in real time.
It classifies facial expressions into the following emotions:

Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral

This project demonstrates how Computer Vision and Deep Learning can be used to interpret human emotions from facial expressions.

Project Overview

Human emotions are an essential part of communication.
This project aims to build an intelligent system that automatically detects emotions from facial expressions.
The model is trained on facial expression images using a Convolutional Neural Network (CNN).

Workflow

- Capture live video from webcam
- Detect faces using Haar Cascade Classifier
- Convert image to grayscale
- Resize face image to 48×48 pixels
- Predict emotion using trained CNN model
- Display predicted emotion on the screen
- Technologies Used

Python
OpenCV
TensorFlow / Keras
NumPy
CNN (Convolutional Neural Network)
Haar Cascade Classifier

Project Structure

face-emotion-detection

images/ → Dataset images
emotiondetector.h5 → Trained model weights
emotiondetector.json → Model architecture
trainmodel.ipynb → Model training notebook
realtimedetection.py → Real-time emotion detection script
requirements.txt → Required Python libraries
README.md → Project documentation

Model Architecture

The emotion classification model is built using a Convolutional Neural Network (CNN).

Typical architecture:

Input Layer (48×48 grayscale image)
Convolution Layer
Activation (ReLU)
Max Pooling
Convolution Layer
Activation (ReLU)
Max Pooling
Flatten Layer
Dense Layer
Dropout
Output Layer (Softmax)
The final output layer predicts 7 emotion classes.

Emotion Classes

0 → Angry
1 → Disgust
2 → Fear
3 → Happy
4 → Sad
5 → Surprise
6 → Neutral

Real-Time Emotion Detection

- The system performs real-time emotion detection using a webcam.

Steps performed:

Webcam captures frame
Face detected using Haar Cascade
Face resized to 48×48 pixels
CNN predicts emotion
Emotion label displayed on screen

Example output:

Happy (96%)
Sad (88%)
Neutral (91%)

Detected emotion appears above the face bounding box.

 Requirements

Main libraries used in this project:
opencv-python
tensorflow
keras
numpy
matplotlib
pandas

Running the Application

- Run the following command to start emotion detection:
- python realtimedetection.py
- The webcam will open and start detecting emotions.
- Press q key to exit the application.

Future Improvements

- This project can be improved by adding features like:
- Emotion confidence percentage display
- Emotion statistics graph
- Emotion history tracking
- Emotion-based music recommendation
- Web application using Flask
- Mobile application integration

Applications

Emotion detection systems can be used in:
- Mental health monitoring
- Smart surveillance systems
- Human-computer interaction
- Customer behavior analysis
- Online education engagement monitoring
- Gaming and entertainment systems

Author
Tushar Agarwal

GitHub:
https://github.com/Tushar-Agrawal-17

Support

If you like this project, consider giving it a Star ⭐ on GitHub.
