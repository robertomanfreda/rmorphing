from __future__ import print_function
import dlib
import cv2 as cv
import sys


PREDICTOR_PATH = '../opencv_utils/predictors/shape_predictor_68_face_landmarks.dat'
FRAME_NAME = 'level_1 - landmark_detector'

faceDetector = dlib.get_frontal_face_detector()
landmarkDetector = dlib.shape_predictor(PREDICTOR_PATH)

def detect_and_display(frame):
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameGray = cv.equalizeHist(frameGray)

    points = fbc.getLandmarks(faceDetector, landmarkDetector, frameGray)

    for point in points:
        cv.circle(frame, point, radius=1, color=(255, 255, 0), thickness=1)

    cv.imshow(FRAME_NAME, frame)


cap = cv.VideoCapture(0)

if not cap.isOpened:
    print('ERROR opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('ERROR - No captured frame!')
        break

    detect_and_display(frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
