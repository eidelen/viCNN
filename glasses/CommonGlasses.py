import cv2

class CvFaceCapture:
    """ This class detects faces in an video stream by using opencv. """

    def __init__(self, capture):
        """
        @param capture: Open cv video capture object
        """
        self.face_detector = cv2.CascadeClassifier(
            '/Users/eidelen/dev/libs/opencv-3.2.0/data/haarcascades/haarcascade_frontalface_default.xml')
        self.capture = capture

    def read(self):
        """ Reads the next video frame and detects the faces on it. """
        ret, frame = self.capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5)
        return frame, faces
