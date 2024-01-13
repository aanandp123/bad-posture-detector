# bad-posture-detector

## Motivation
Sitting in front of a laptop for long hours, I found myself spending hours hunched over my laptop, unaware of how it affected my posture.

I was experiencing the consequences of poor posture – backaches, neck pain, etc. 


## Overview

This repository contains a posture detection system implemented using a Convolutional Neural Network (CNN). The CNN is trained to classify good and bad postures based on images captured in real-time through a webcam. The model achieves an accuracy of around 91% and an F1 score of approximately 0.86.

## Features

- **Real-time Posture Analysis**: Uses the laptop's webcam to provide real-time analysis of the user's posture.

- **Pose Landmarks**: It tracks key points on the user's body to make a comprehensive assessment of their posture.

- **Feedback**: Instantly notifies the user whether they have good or bad posture, allowing them to make immediate corrections.

- **Data Analytics**: After exit, graphs will be displayed showing the percentage of time spent in bad and good posture and 
time periods with corresponding posture classifications in a line graph.

## Usage

1. **Clone the Repository**: Start by cloning this repository to your local machine:
2. **Get started with data and model**:To train the model locally, add the posture data from the Google Drive link to the specified folder and run model.py. Alternatively, obtain the pre-trained model.keras file from the provided Google Drive link.
3. **Run**: Execute the file main.py
4. **Exit**: To exit, press 'q' on the keyboard. Data analytics will be displayed.


