from __future__ import print_function
import cv2 as cv
import sys

current_file_name = sys.argv[0]
FRAME_NAME = current_file_name.split('.')[0]

def detect_and_display(frame):
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
