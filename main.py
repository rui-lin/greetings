import cv2
import numpy as np

# todo - speedup.
DEBUG = True

class FaceDetector:
    def __init__(self):
        path = "data/haarcascade_frontalface_default.xml"
        #path = "data/haarcascade_frontalface_alt_tree.xml"
        #path = "data/lbpcascade_frontalface.xml"
        #path = "data/haarcascade_upperbody.xml"
        self.faceCascade = cv2.CascadeClassifier(path)

    # returns array of (x,y,w,h) of faces
    # img should be grayscale
    def detect_faces(self, img):
        faces = self.faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(30, 30),
            flags=0#cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        return faces

class StateEnum:
    NOT_CALIBRATED, CALIBRATING, CALIBRATED = range(3)

class MovingObjectDetector:
    def __init__(self):
        self.history = 50
        self.nmixtures = 3
        self.backgroundRatio = 0.000000001 # how much is background
        self.bgsubtractor = None
        self.learn_counter = 0
        self.state = StateEnum.NOT_CALIBRATED

    # Start calibration again.
    def start_calibration(self):
        self.state = StateEnum.CALIBRATING
        self.learn_counter = 0
        self.bgsubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
            backgroundRatio=self.backgroundRatio, 
            nmixtures=self.nmixtures, 
            history=self.history
        )
        print "starting - moving objector detector calibration"

    def detect_moving_objects(self, img):
        # Do state changes
        if self.learn_counter >= self.history:
            self.state = StateEnum.CALIBRATED
            self.learn_counter = 0
            print "completed - moving objector detector calibration"

        # Do state-based processing
        if self.state == StateEnum.CALIBRATING:
            masked_img = self.bgsubtractor.apply(img, learningRate = 1.0/self.history) # edit
            self.learn_counter += 1

            if DEBUG:
                cv2.imshow('moving_object_detector_debug', masked_img)

            _unused_img, contours, hierarchy = cv2.findContours(
                masked_img,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [cnt for cnt in contours if len(cnt)>0 and cv2.contourArea(cnt) > 50*50]
            return contours
        elif self.state == StateEnum.CALIBRATED:
            masked_img = self.bgsubtractor.apply(img, learningRate = 0)

            if DEBUG:
                cv2.imshow('moving_object_detector_debug', masked_img)

            _unused_img, contours, hierarchy = cv2.findContours(
                masked_img,
                mode=cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 50*50]
            return contours
        else:
            print "WARNING: uncalibrated usage of moving object detector!"
            return [[]]




"""
    Overview of architecture.
    The main method handles grabbing frames and drawing.

    Each support module will have:
        - __init__ called upon creation,
        - update called every frame.
"""

def main():
    cap = cv2.VideoCapture(0)
    faceDetector = FaceDetector()
    objDetector = MovingObjectDetector()
    objDetector.start_calibration()

    frame_counter = 0
    while ( cap.isOpened() ):
        ret, frame = cap.read()

        contours = objDetector.detect_moving_objects(frame)

        if frame_counter % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceDetector.detect_faces(gray)

        # Draw moving objects contour
        if len(contours) > 0:
            for cnt in contours:
                cv2.drawContours(frame,[cnt],0,(0,255,0),2)

        # Draw around face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        cv2.imshow('img', frame)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
        if k == ord('c'):
            objDetector.start_calibration()

        frame_counter = (frame_counter + 1) % 10

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()