import cv2
import numpy as np
from datetime import datetime
from keras.models import load_model
from pose_util import mpPose, preprocess_image
from pose_analytics import analyze_posture

# Load the trained model
model = load_model('model.keras')

# Body landmarks to focus on
keyPoints = [9,10,11,12,13,14,15,16]

# Set the webcam parameters
width = 1280
height = 720 
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Initialize the mpPose class  
findPose = mpPose()

# Smoothing parameters
smoothing_factor = 0.2
previous_prediction = None

# Data for analytics
results = {'Timestamp': [], 'Posture': []}

while True:
    # Read the frame
    _, frame = camera.read()
    
    # Process the frame using pose detection
    poseData = findPose.Marks(frame)
    
    # Preprocess the frame for the CNN
    preprocessed_frame = preprocess_image(frame)
    
    # Reshape the frame to match the input shape expected by the model
    input_frame = np.expand_dims(preprocessed_frame, axis=0)
    
    # Make predictions using the trained model
    prediction = model.predict(input_frame)
    
    # Apply smoothing so predictions are not so erratic
    if previous_prediction is not None:
        prediction = smoothing_factor * prediction + (1 - smoothing_factor) * previous_prediction

    # Convert the prediction to a 0 or 1 based threshold
    predicted_class = 1 if prediction > 0.5 else 0
    
    # Record timestamp and predicted posture
    timestamp = datetime.now()
    results['Timestamp'].append(timestamp)
    results['Posture'].append(predicted_class)

    # Display the prediction on the frame
    if predicted_class == 1:
        cv2.putText(frame, "Good Posture", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 9)
    else:
        cv2.putText(frame, "Bad Posture", (100, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (0,0,255), 9)

    # Display the key points
    if len(poseData)!=0:
        for ind in keyPoints:
            cv2.circle(frame, poseData[ind], 15, (0, 255, 0), 3)
    # Display the frame
    cv2.imshow('Posture Classification', frame)
    

    # Update the previous prediction
    previous_prediction = prediction

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
camera.release()
cv2.destroyAllWindows()

# Perform data analytics
analyze_posture(results)
