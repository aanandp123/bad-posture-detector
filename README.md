# bad-posture-detector


Sitting in front of a laptop for long hours, I found myself spending hours hunched over my laptop, unaware of how it affected my posture.

I was experiencing the consequences of poor posture – backaches, neck pain, etc. 


## Overview

This program is designed to detect deviations in a person's posture using the laptop's webcam. It leverages the Mediapipe library to analyze the pose of an individual in real-time and determine whether their posture is good or bad. It provides instant feedback to the user and keeps track of instances of bad posture.

## Features

- **Real-time Posture Analysis**: The program uses the laptop's webcam to provide real-time analysis of the user's posture.

- **Pose Landmarks**: It tracks key points on the user's body to make a comprehensive assessment of their posture.

- **Feedback**: The program instantly notifies the user whether they have good or bad posture, allowing them to make immediate corrections.

- **Bad Posture Count**: It keeps track of instances of bad posture, helping users monitor their posture improvement over time.

## Usage

1. **Clone the Repository**: Start by cloning this repository to your local machine:
2. **Navigate to the Project Directory**: Change into the project directory:
3. **Run the Program**: Execute the program
4. **Follow On-Screen Instructions**:
   - Initially, show your correct posture in front of the camera and press 't' on the keyboard when you are ready to train the model.
   - After training, the program continuously analyzes your posture and displays "Good Posture" or "Bad Posture" accordingly.
   - The program also counts instances of bad posture.
5. **Exit the Program**: To exit the program, press 'z' on the keyboard.


## Customization

You can customize the behavior of the program by adjusting the following parameters in the `posture_detection.py` file:

- `still`: Set to `True` for still image mode or `False` for real-time webcam mode.
- `upperBody`: Set to `True` to track only upper body keypoints, or `False` for full-body tracking.
- `smoothData`: Set to `True` to enable smoothing of pose landmarks.
- `tol1` and `tol2`: Adjust the minimum confidence thresholds for detection and tracking.

