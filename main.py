import cv2
import numpy as np
import datetime
import glob
import os

DEBUG = True

class FaceDetector:
    def __init__(self):
        #path = "data/haarcascade_frontalface_default.xml"
        path = "data/haarcascade_frontalface_alt.xml"
        #path = "data/lbpcascade_frontalface.xml"
        #path = "data/haarcascade_upperbody.xml"
        self.faceCascade = cv2.CascadeClassifier(path)

    # returns array of (x,y,w,h) of faces
    # img should be grayscale
    def detect_faces(self, img):
        faces = self.faceCascade.detectMultiScale(
            img,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50),
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

class FaceRecognizer:
    """ 
    LBPH over eigenfaces + fisherfaces, primarily because it can be updated
    """
    def __init__(self):
        self.model = cv2.face.createLBPHFaceRecognizer()
        self.max_training_counter = 5
        self.training_counter = 0
        self.training_label = None
        self.state_before_training = StateEnum.NOT_CALIBRATED
        self.state = StateEnum.NOT_CALIBRATED
        self.loaded_file = None


    def start_training(self, label):
        self.state_before_training = self.state
        self.state = StateEnum.CALIBRATING
        self.training_counter = 0
        self.training_label = label
        print "start updating face %i" % self.training_label

    # Returns list of (face, label, confidence) for each face given
    def recognize_faces(self, img, faces):
        # State changes
        if self.training_counter >= self.max_training_counter:
            self.state = StateEnum.CALIBRATED
            print "updated face %i" % self.training_label
            self.training_label = None
            self.training_counter = 0

        # State based processing
        if self.state == StateEnum.CALIBRATING:
            if len(faces) == 1:
                images = [crop_image(img, faces[0])]
                labels = np.array([self.training_label])
                self.model.update(images, labels)
                self.training_counter += 1
            else:
                print "WARNING: multiple faces detected - canceling training"
                self.training_label = None
                self.training_counter = 0
                self.state = self.state_before_training
            return []
        elif self.state == StateEnum.CALIBRATED:
            # note: the + is a concat tuple operation
            predicts = [(face,) + self.model.predict(crop_image(img, face)) for face in faces]
            return predicts
        elif self.state == StateEnum.NOT_CALIBRATED:
            return []
        else:
            assert False # Unexpected state

    # newfile is whether to save to a new file
    def save(self, newfile=False):
        if not self.loaded_file: # no loaded file, create new
            newfile=True

        if newfile or not self.loaded_file:
            filename = "local/face_model_" + datetime.datetime.now().isoformat() + ".yaml"
            filename = filename.replace(":",".")
            self.loaded_file = filename
            print "saved new model snapshot to", self.loaded_file
        else:
            print "updated model", self.loaded_file
        self.model.save(self.loaded_file)

    def load_newest_model(self):
        model_files = list(glob.iglob('local/face_model_*.yaml'))
        if any(model_files):
            newest_model_file = max(model_files, key=os.path.getctime)
            print "loading existing facial recognition data from %s" % newest_model_file
            self.model.load(newest_model_file)
            self.state = StateEnum.CALIBRATED
            self.loaded_file = newest_model_file


# Rect should be a 4-tuple (x, y, w, h)
def crop_image(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]


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
    #objDetector = MovingObjectDetector()
    faceRecognizer = FaceRecognizer()
    faceRecognizer.load_newest_model()
    #objDetector.start_calibration()
    label_map = {
        1:'Rui',
        2:'Jessica',
        3:'Mom',
        4:'Dad'
    }

    frame_counter = 0
    while ( cap.isOpened() ):
        ret, frame = cap.read()

        #contours = objDetector.detect_moving_objects(frame)

        if frame_counter % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceDetector.detect_faces(gray)
            recognized_faces = faceRecognizer.recognize_faces(gray, faces)

        # Draw moving objects contour
        #if len(contours) > 0:
        #    for cnt in contours:
        #        cv2.drawContours(frame,[cnt],0,(0,255,0),2)

        # Draw around face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw face recognition annotations
        for (face, label, unconfidence) in recognized_faces:
            if unconfidence >= 50:
                text = "%s ? (var: %i)" % (label_map[label], unconfidence)
            else:
                text = "%s (var: %i)" % (label_map[label], unconfidence)

            x,y,w,h = face
            cv2.putText(
                frame, 
                text = text,
                org = (max(x-10,0), max(y-10,0)), 
                fontFace = cv2.FONT_HERSHEY_PLAIN,
                fontScale = 1.0,
                color = (0,255,0),
                thickness = 2
            )
        
        cv2.imshow('img', frame)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            faceRecognizer.save()
            break
        #if k == ord('c'):
        #    objDetector.start_calibration()
        if k == ord('1'):
            faceRecognizer.start_training(1)
        if k == ord('2'):
            faceRecognizer.start_training(2)
        if k == ord('3'):
            faceRecognizer.start_training(3)
        if k == ord('4'):
            faceRecognizer.start_training(4)
        if k == ord('s'): # save new snapshot
            faceRecognizer.save(newfile=True)

        frame_counter = (frame_counter + 1) % 10

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()