import cv2
import mediapipe as mp
width = 1280
height = 720 

class mpPose:
    import mediapipe as mp
    def __init__(self, still=False, upperBody=False, smoothData=True, tol1=0.5, tol2=0.5):
        # Initialize the Pose model with the parameters
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


def preprocess_image(img):
    # Resize the image
    img = cv2.resize(img, (224, 224))
    # Normalize pixel values so that they're between 0 and 1
    img = img / 255.0
    return img
