from __future__ import print_function
from imutils import face_utils
import dlib
import cv2 as cv

PREDICTOR_PATH = '../opencv_utils/predictors/shape_predictor_68_face_landmarks.dat'
FRAME_NAME = 'level_1 - 2_landmark_detector_fps'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def detect_and_display(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    rects = detector(frame_gray, 1)

    for (i, rect) in enumerate(rects):
        shape = predictor(frame_gray, rect)
        shape = face_utils.shape_to_np(shape)
        for (x, y) in shape:
            cv.circle(frame, (x, y), 1, (0, 0, 255), -1)

    fps = str(cap.get(cv.CAP_PROP_FPS))
    cv.putText(frame, fps, (7, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (100, 255, 0), 3, cv.LINE_AA)
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
