
import certifi
import ssl
# Set SSL certificate path
ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_https_context().load_verify_locations(certifi.where())

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from keras import layers, models
from pose_util import mpPose, preprocess_image

width = 1280
height = 720 

# Initialize the mpPose class for pose detection
findPose = mpPose()

# Placeholder for storing training data
training_data = []

# Path to the directory containing training images for good posture
good_posture_directory = 'posturedata/good'

# Iterate over files in the good posture training directory
for filename in os.listdir(good_posture_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        file_path = os.path.join(good_posture_directory, filename)
        
        # Read the image
        img = cv2.imread(file_path)
        
        # Process the image using pose detection and other steps as needed
        poseData = findPose.Marks(img)
        
        # Preprocess the image and add it to the training data
        preprocessed_img = preprocess_image(img)
        
        # Assign label 1 for good posture
        label = 1
        
        training_data.append((preprocessed_img, label))

# Path to the directory containing training images for bad posture
bad_posture_directory = 'posturedata/bad'

# Iterate over files in the bad posture training directory
for filename in os.listdir(bad_posture_directory):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        file_path = os.path.join(bad_posture_directory, filename)
        
        # Read the image
        img = cv2.imread(file_path)
        
        # Process the image using pose detection and other steps as needed
        poseData = findPose.Marks(img)
        
        # Preprocess the image and add it to the training data
        preprocessed_img = preprocess_image(img)
        
        # Assign label 0 for bad posture
        label = 0
        
        training_data.append((preprocessed_img, label))

# Separate the data into features (X) and labels (y)
X = np.array([item[0] for item in training_data])
y = np.array([item[1] for item in training_data])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Separate the data into features (X) and labels (y)
X = np.array([item[0] for item in training_data])
y = np.array([item[1] for item in training_data])

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple convolutional neural network (CNN)
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification (good or bad posture)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")
y_pred = model.predict(X_test)

y_pred_classes = (y_pred > 0.5).astype(int) 
f1 = f1_score(y_test, y_pred_classes)

print(f"F1 Score: {f1}")

model.save('model.keras')