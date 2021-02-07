from __future__ import print_function
import dlib
import cv2 as cv
from imutils import face_utils
import sys


PREDICTOR_PATH = '../opencv_utils/predictors/shape_predictor_68_face_landmarks.dat'
FRAME_NAME = 'level_1 - landmark_detector'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def detect_and_display(frame):
    frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frameGray = cv.equalizeHist(frameGray)

    rects = detector(frameGray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(frameGray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, "Face #{}".format(i + 1), (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        for (x, y) in shape:
            cv.circle(frame, (x, y), 1, (0, 0, 255), -1)

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
