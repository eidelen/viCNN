"""
This module reads a video input and saves a rectangular
face patch as a jpg file.
saving -> press 'a'
fix rectangle position -> press 'f'
exit -> press 'q'
"""

import cv2
import time
from CommonGlasses import CvFaceCapture

cv_video_input = cv2.VideoCapture(0)
capture = CvFaceCapture(cv_video_input)
output_path = 'data/current/'

addFaceToFile = False
fixRect = False
count = 0
x, y, w, h = 0, 0, 5, 5

while(True):
    frame, faces = capture.read()
    if len(faces) > 0 or fixRect:

        # update the rect pos with the actual location of the current face
        if not fixRect:
            [x, y, w, h] = faces[0]

        rect_color = (255, 0, 0)
        if addFaceToFile:
            # save the current face image
            face = frame[y:y+h, x:x+w]
            file_name = output_path + str(time.time()*1000) + ".jpg"
            cv2.imwrite(file_name, face)
            count += 1
            addFaceToFile = False
            rect_color = (0, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Number of samples ' + str(count), (40, 40), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    keyCode = cv2.waitKey(100)
    if keyCode == ord('q'):
        break
    if keyCode == ord('a'):
        addFaceToFile = not addFaceToFile
    if keyCode == ord('f'):
        fixRect = not fixRect

cv_video_input.release()
cv2.destroyAllWindows()
