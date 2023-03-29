import cv2
import mediapipe as mp
import numpy as np
import time


class mpPose:
    import mediapipe as mp

    def __init__(self, still=False, upperBody=False, smoothData=True, tol1=0.5, tol2=0.5 ):
        self.myPose = self.mp.solutions.pose.Pose(still,upperBody,smoothData, min_detection_confidence =tol1, min_tracking_confidence=tol2)
    
    def Marks(self,frame):
        frameRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results = self.myPose.process(frameRGB)
        poseLandmarks=[]
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                poseLandmarks.append((int(lm.x*width) ,int(lm.y*height)))
        return poseLandmarks   


def findDistances(poseData):
    distMatrix = np.zeros([len(poseData),len(poseData)], dtype = 'float')
    shoulderWidth = ((poseData[11][0]-poseData[12][0])**2 + (poseData[11][1]-poseData[12][1])**2)**(1.0/2.0)
    for row in range(0, len(poseData)):
        for col in range(0, len(poseData)):
            distMatrix[row][col] = (((poseData[row][0]-poseData[col][0])**2 + (poseData[row][1]-poseData[col][1])**2)**(1.0/2.0))/shoulderWidth
    return distMatrix

def findError(goodPostureMatrix,unknownMatrix,keyPoints):
    error = 0
    for row in keyPoints:
        for col in keyPoints:
            error = error + abs(goodPostureMatrix[row][col] - unknownMatrix[row][col])
    return error

start_time = time.time()
badposturecount = 0
lastframebad = False
def determinePosture(goodPostureMatrix,unknownMatrix,keyPoints,tol):
    global start_time
    global badposturecount
    global lastframebad
    error = findError(goodPostureMatrix,unknownMatrix,keyPoints)
    if error <= tol:
        if lastframebad:
            
            end_time = time.time()
            
            time_elapsed = end_time - start_time
            if time_elapsed > 5:

                badposturecount = badposturecount + 1
            lastframebad = False
        return "Good Posture"
    else: #bad posture
        if not lastframebad:
            start_time = time.time()
            
            lastframebad = True
        return "Bad Posture"


width = 1280
height = 720 
camera = cv2.VideoCapture(0) #cv2.CAP_DSHOW inside brakcets doesnt work
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
camera.set(cv2.CAP_PROP_FPS, 30)
camera.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))

findPose = mpPose()

keyPoints = [9,10,11,12,13,14,15,16]

train = True
tol = 5

while True:
    _, frm = camera.read()
    
    poseData = findPose.Marks(frm)

    if train:
        if len(poseData)!=0:
            print('Show your correct posture, press t on keyboard when ready')
            if cv2.waitKey(1) ==ord('t'):
                goodPostureMatrix = findDistances(poseData)
                train = False

    if train!=True:
        if len(poseData)!=0:
            unknownMatrix = findDistances(poseData)
            #error = findError(goodPostureMatrix, unknownMatrix, keyPoints)
            text = determinePosture(goodPostureMatrix,unknownMatrix, keyPoints,tol)
            if text == "Good Posture":
                cv2.putText(frm,text,(100,100),cv2.FONT_HERSHEY_COMPLEX,4,(0,255,0),9)
            else:
                cv2.putText(frm,text,(100,100),cv2.FONT_HERSHEY_COMPLEX,4,(0,0,255),9)

            cv2.putText(frm,str(badposturecount),(200,200),cv2.FONT_HERSHEY_COMPLEX,4,(100,100,0),9)    

    
    if len(poseData)!=0:
        for ind in keyPoints:
            cv2.circle(frm,poseData[ind],15,(0,255,0),3)

    cv2.imshow('Mac camera', frm)
    cv2.moveWindow('Mac camera', 100,0)
    if cv2.waitKey(1) ==ord('z'):
        break
camera.release()