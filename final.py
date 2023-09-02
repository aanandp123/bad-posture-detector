import cv2
import mediapipe as mp
import numpy as np
import time

# Define a class for using the Mediapipe Pose model
class mpPose:
    import mediapipe as mp

    def __init__(self, still=False, upperBody=False, smoothData=True, tol1=0.5, tol2=0.5 ):
        # Initialize the Pose model with specified parameters
        self.myPose = self.mp.solutions.pose.Pose(
            still, upperBody, smoothData, min_detection_confidence=tol1, min_tracking_confidence=tol2)

    def Marks(self, frame):
        # Process the input frame and extract pose landmarks
        frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.myPose.process(frameRGB)
        poseLandmarks = []
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                poseLandmarks.append((int(lm.x * width), int(lm.y * height)))
        return poseLandmarks

# Function to calculate distance matrix between pose landmarks
def findDistances(poseData):
    distMatrix = np.zeros([len(poseData), len(poseData)], dtype='float')
    shoulderWidth = ((poseData[11][0] - poseData[12][0]) ** 2 + (poseData[11][1] - poseData[12][1]) ** 2) ** (1.0/2.0)
    for row in range(0, len(poseData)):
        for col in range(0, len(poseData)):
            # Calculate the distance normalized to shoulder width
            distMatrix[row][col] = (((poseData[row][0] - poseData[col][0]) ** 2 + 
                                     (poseData[row][1] - poseData[col][1]) ** 2) ** (1.0/2.0)) / shoulderWidth
    return distMatrix

# Function to calculate error in posture
def findError(goodPostureMatrix, unknownMatrix, keyPoints):
    error = 0
    for row in keyPoints:
        for col in keyPoints:
            # Calculate the absolute difference between good and unknown posture matrices
            error = error + abs(goodPostureMatrix[row][col] - unknownMatrix[row][col])
    return error

# Initialize variables for posture detection
start_time = time.time()
badposturecount = 0
lastframebad = False

# Function to determine posture (Good or Bad)
def determinePosture(goodPostureMatrix, unknownMatrix, keyPoints, tol):
    global start_time
    global badposturecount
    global lastframebad
    error = findError(goodPostureMatrix, unknownMatrix, keyPoints)
    if error <= tol:
        if lastframebad:
            end_time = time.time()
            time_elapsed = end_time - start_time
            # If bad posture continues for more than 5 seconds, count it as a bad posture instance
            if time_elapsed > 5:
                badposturecount = badposturecount + 1
            lastframebad = False
        return "Good Posture"
    else:  # Bad posture
        if not lastframebad:
            start_time = time.time()
            lastframebad = True
        return "Bad Posture"

# Set the webcam capture parameters
width = 1280
height = 720 
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

# Initialize the mpPose class for pose detection
findPose = mpPose()

# Define key points for posture analysis
keyPoints = [9, 10, 11, 12, 13, 14, 15, 16]

# Training mode flag and tolerance for posture comparison
train = True
tol = 5

while True:
    _, frm = camera.read()
    poseData = findPose.Marks(frm)

    if train:
        if len(poseData) != 0:
            print('Show your correct posture, press "t" on the keyboard when ready')
            if cv2.waitKey(1) == ord('t'):
                goodPostureMatrix = findDistances(poseData)
                train = False

    if not train:
        if len(poseData) != 0:
            unknownMatrix = findDistances(poseData)
            text = determinePosture(goodPostureMatrix, unknownMatrix, keyPoints, tol)
            if text == "Good Posture":
                cv2.putText(frm, text, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 255, 0), 9)
            else:
                cv2.putText(frm, text, (100, 100), cv2.FONT_HERSHEY_COMPLEX, 4, (0, 0, 255), 9)

            cv2.putText(frm, str(badposturecount), (200, 200), cv2.FONT_HERSHEY_COMPLEX, 4, (100, 100, 0), 9)

    if len(poseData) != 0:
        for ind in keyPoints:
            cv2.circle(frm, poseData[ind], 15, (0, 255, 0), 3)

    cv2.imshow('Laptop Camera', frm)
    cv2.moveWindow('Laptop Camera', 100, 0)
    
    if cv2.waitKey(1) == ord('z'):
        break

camera.release()
