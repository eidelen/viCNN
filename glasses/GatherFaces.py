import cv2
import time


#capture = cv2.VideoCapture('/Users/eidelen/dev/viCNN/glasses/data/input/nari_adi.mov')
capture = cv2.VideoCapture(0)
output_path = '/Users/eidelen/dev/viCNN/glasses/data/current/'

face_cascade = cv2.CascadeClassifier('/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
eyes_cascade = cv2.CascadeClassifier('/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_eye.xml')

addFaceToFile = False
fixRect = False
count = 0
eyeDistList = []
x, y, w, h = 0, 0, 5, 5

while(True):

    ret, frame = capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # find face
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0 or fixRect:

        if not fixRect:
            [x, y, w, h] = faces[0] # update the rect pos with the current face

        if addFaceToFile:
            # save face image
            face = frame[y:y+h, x:x+w]
            file_name = output_path + str(time.time()*1000) + ".jpg"
            cv2.imwrite(file_name, face)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
            count += 1
            addFaceToFile = False
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 4)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Number of samples ' + str(count), (40, 40), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame', frame)

    keyCode = cv2.waitKey(100)
    if keyCode == ord('q'):
        break;
    if keyCode == ord('a'):
        addFaceToFile = not addFaceToFile
    if keyCode == ord('f'):
        fixRect = not fixRect

capture.release()
cv2.destroyAllWindows()
