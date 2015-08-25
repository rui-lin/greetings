import cv2
import numpy as np
import datetime
import glob
import os
import subprocess
from threading import Thread
import Queue

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
            minSize=(20, 20),
            flags=0#cv2.cv.CV_HAAR_SCALE_IMAGE
        )
        return faces

class StateEnum:
    NOT_CALIBRATED, CALIBRATING, CALIBRATED = range(3)

class StaticBackgroundSubtractor:
    def __init__(self):
        self.background = None
        self.kernel_xlarge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15))
        self.kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        self.kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

    def apply(self, img, learningRate):
        img = cv2.medianBlur(img, 5) # todo decide if needed.

        if learningRate == 1:
            self.background = img

        # Get foreground mask
        diff = cv2.absdiff(self.background, img)
        _ret, diff = cv2.threshold(diff, 20, 240, cv2.THRESH_BINARY)
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, self.kernel_small)

        # Get loose foreground mask
        foreground_mask = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.kernel_large)
        _ret, foreground_mask = cv2.threshold(foreground_mask, 1, 255, cv2.THRESH_BINARY)
        masked_img = cv2.bitwise_and(img, img, mask=foreground_mask)

        cv2.imshow('foreground mask', diff)
        cv2.imshow('masked_img', masked_img)
        cv2.moveWindow('foreground mask', 0, 0)

        # Get edges in loose foreground
        edges = cv2.Canny(masked_img, 30, 60, apertureSize=3, L2gradient=False) # try diff of edges
        edges = cv2.dilate(edges, None, iterations=3)

        cv2.imshow('canny', edges)
        cv2.moveWindow('canny', 0, 500)
        
        _ret, edges = cv2.threshold(edges, 1, 15, cv2.THRESH_BINARY) # edge alone can't make something

        # Add foreground + edges from loose foreground
        diff = diff + edges

        # Close, expand whites
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.kernel_large)

        # Detect embedded contours, fill in
        _unused, contours, hier = cv2.findContours(diff, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            hier = hier[0] # randomly one extra nest..

        for i, cnt in enumerate(contours):
            (nxt, prev, child, par) = hier[i]
            #if par >= 0: # has parent
            if cv2.contourArea(cnt) < 50*50: # too small, suppress
                cv2.drawContours(diff, [cnt], 0, 0, -1)
            else:
                cv2.drawContours(diff, [cnt], 0, 255, -1)

        # Return foreground
        _ret, diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        return diff


class MovingObjectDetector:
    def __init__(self):
        self.history = 60
        self.nmixtures = 3
        self.backgroundRatio = 0.1 # how much is background
        self.bgsubtractor = None
        self.learn_counter = 0
        self.state = StateEnum.NOT_CALIBRATED

    # Start calibration again.
    def start_calibration(self):
        self.state = StateEnum.CALIBRATING
        self.learn_counter = 0
        """
        self.bgsubtractor = cv2.bgsegm.createBackgroundSubtractorMOG(
            backgroundRatio=self.backgroundRatio, 
            nmixtures=self.nmixtures, 
            history=self.history
        )
        """
        """
        self.bgsubtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=32  # threshold for distance, to see whether same vs new component
                             # higher value reduce sensitivity
        )
        self.bgsubtractor.setBackgroundRatio(self.backgroundRatio)
        self.bgsubtractor.setDetectShadows(False)
        self.bgsubtractor.setComplexityReductionThreshold(0.001) # 0.05 is default
        self.bgsubtractor.setVarThresholdGen(100)
        """
        self.history = 1
        self.bgsubtractor = StaticBackgroundSubtractor()
        """
        self.bgsubtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
            initializationFrames=self.history,
            decisionThreshold=self.backgroundRatio
        )
        """
        """
        self.bgsubtractor = cv2.createBackgroundSubtractorKNN()
        self.bgsubtractor.setHistory(self.history)
        """
        self.bgkernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

        print "starting - moving objector detector calibration"

    def detect_moving_objects(self, img):
        # Do state changes
        if self.learn_counter >= self.history:
            self.state = StateEnum.CALIBRATED
            self.learn_counter = 0
            print "completed - moving objector detector calibration"

        # Do state-based processing
        if self.state == StateEnum.CALIBRATING:
            masked_img = self.bgsubtractor.apply(img, learningRate = 1/self.history) # edit
            #masked_img = cv2.erode(masked_img, None, iterations=2)
            #masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE, self.bgkernel)
            masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_OPEN, self.bgkernel)
            self.learn_counter += 1

            if DEBUG:
                cv2.imshow('moving_object_detector_debug', masked_img)
                cv2.moveWindow('moving_object_detector_debug', 700, 500)

            _unused_img, contours, hierarchy = cv2.findContours(
                masked_img,
                mode=cv2.RETR_TREE, #cv2.RETR_FLOODFILL, 
                method=cv2.CHAIN_APPROX_SIMPLE
            )
            contours = [cnt for cnt in contours if len(cnt)>0 and cv2.contourArea(cnt) > 50*50]
            return contours
        elif self.state == StateEnum.CALIBRATED:
            masked_img = self.bgsubtractor.apply(img,  learningRate = 0.1/self.history)
            #masked_img = cv2.erode(masked_img, None, iterations=2)
            #masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_CLOSE, self.bgkernel)
            masked_img = cv2.morphologyEx(masked_img, cv2.MORPH_OPEN, self.bgkernel)

            if DEBUG:
                cv2.imshow('moving_object_detector_debug', masked_img)
                cv2.moveWindow('moving_object_detector_debug', 700, 500)

            _unused_img, contours, hierarchy = cv2.findContours(
                masked_img,
                mode=cv2.RETR_TREE, #cv2.RETR_FLOODFILL, 
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
    Note: Each model.getHistogram() is a datapoint
    """
    def __init__(self):
        self.model = cv2.face.createLBPHFaceRecognizer()

        self.max_training_counter = 5
        self.training_counter = 0
        self.training_label = None
        self.state_before_training = StateEnum.NOT_CALIBRATED
        self.state = StateEnum.NOT_CALIBRATED
        self.loaded_file = None

    """
    Starts training a new label(string)
    This method will find a int label for it, or create one if new.
    """
    def start_training(self, string_label):
        # Find int_label for string_label
        int_label = self.model.getLabelsByString(string_label)
        int_label = [i for i in int_label if self.model.getLabelInfo(i) == string_label]
        if len(int_label) > 1:
            print "ERROR: more than one int label with string '%s'. Aborting training." % string_label
            return
        elif len(int_label) == 1:
            int_label = int_label[0]
        elif len(int_label) == 0:   # no labels found, create new
            used_ints = self.model.getLabels()
            if used_ints is None:
                used_ints = []
            try:
                int_label = (i for i in xrange(1,2**62) if i not in used_ints).next()
                self.model.setLabelInfo(int_label, string_label)
            except:
                print "ERROR: attempt to exceed maximum of 2**62 labels. Aborting training."
                return

        self.state_before_training = self.state
        self.state = StateEnum.CALIBRATING
        self.training_counter = 0
        self.training_label = int_label
        print "start updating face %i (%s)" % (self.training_label, string_label)

    # Finds next available name in format [name]-[next_avail_int_counter].[extension]
    def find_next_training_img_name(self, label, extension):
        prefix = label.replace(" ", "_")
        for i in xrange(1,2**62):
            name = 'local/%s-%i.%s' % (prefix, i, extension)
            if not any(glob.iglob(name)):
                return name
        return None


    # Returns list of (face, label, confidence) for each face given
    def recognize(self, img, faces):
        # State changes
        if self.training_counter >= self.max_training_counter:
            self.state = StateEnum.CALIBRATED
            print "done updating face %i." % self.training_label
            self.training_label = None
            self.training_counter = 0

        # State based processing
        if self.state == StateEnum.CALIBRATING:
            if len(faces) == 1:
                images = [crop_image(img, faces[0])]
                labels = np.array([self.training_label])
                self.model.update(images, labels)
                self.training_counter += 1

                filename = self.find_next_training_img_name(
                    self.model.getLabelInfo(self.training_label),
                    "png"
                )

                cv2.imwrite(filename, images[0])
            else:
                print "WARNING: multiple faces detected - canceling training"
                self.training_label = None
                self.training_counter = 0
                self.state = self.state_before_training
            return []
        elif self.state == StateEnum.CALIBRATED:
            predicts = []
            for face in faces:
                (label, var) = self.model.predict(crop_image(img, face))
                predicts += [(face, self.model.getLabelInfo(label), var)]
            return predicts
        elif self.state == StateEnum.NOT_CALIBRATED:
            return []
        else:
            assert False # Unexpected state

    # newfile is whether to save to a new file
    def save(self, newfile=False):
        if not self.loaded_file: # no loaded file, create new
            newfile = True

        if newfile:
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
            print "Loading existing facial recognition data from %s .." % newest_model_file
            self.model.load(newest_model_file)
            self.state = StateEnum.CALIBRATED
            self.loaded_file = newest_model_file
            print "Done."


# Rect should be a 4-tuple (x, y, w, h)
def crop_image(img, rect):
    x, y, w, h = rect
    return img[y:y+h, x:x+w]

# Returns bool, if rect r is inside q.
def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh


"""
    Overview of architecture.
    The main method handles grabbing frames and drawing.

    Each support module will have:
        - __init__ called upon creation,
        - update called every frame.
"""

class SpeechModule:
    def __init__(self):
        self.acknowledged = set()
        self.t1 = Thread(target = self.run_1) 
        self.t1.daemon = True # daemon so will not block exiting. 
                              # NB daemon threads shouldn't hold any sys resources (eg. write file)
        self.speech_queue = Queue.Queue()

    def run_1(self):
        while True:
            # Speak one sentence at a time.
            sentence = self.speech_queue.get(block=True)
            #proc = subprocess.Popen(["espeak", sentence])
            #proc.wait()

    def start(self):
        self.t1.start()

    def update_faces_in_view(self, faces):
        # If people move out of view, they are not acknowledged anymore.
        current = set(label for (_, label, _) in faces)
        self.acknowledged = self.acknowledged & current

        # See if there's new faces to acknowledge.
        for (face, label, unconfidence) in faces:
            if unconfidence < 150 and label not in self.acknowledged:
                self.say_hello(label)
                self.acknowledged.add(label)

    def say_hello(self, person):
        if person == "Rui Lin":
            self.speech_queue.put("Hello master Ray.")
        elif person == "Jessica":
            self.speech_queue.put("Hello master Jessica.")
        else:
            self.speech_queue.put("Hello %s." % person)

    def say_goodbye(self, person):
        if person == "Rui Lin":
            self.speech_queue.put("Goodbye master Ray.")
        elif person == "Jessica":
            self.speech_queue.put("Goodbye master Jessica.")
        else:
            self.speech_queue.put("Goodbye %s." % person)


class BodyDetector:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Returns list of rect-tuples (rx, ry, rw, rh)
    def detect_bodies(self, img):
        found, weights = self.hog.detectMultiScale(img, 
            winStride=(8,8), padding=(32,32), scale=1.05)
        found_filtered = []
        for ri, r in enumerate(found):
            for qi, q in enumerate(found):
                if ri != qi and inside(r, q):
                    break
            else:
                found_filtered.append(r)
        # todo look into 'useMeanshiftGrouping'
        return found_filtered


class Object:
    def __init__(self, contour=[], centroid=(), id=None, name=None, updated=True):
        self.contour = contour
        self.centroid = centroid
        self.id = id
        self.name = name
        self.updated = updated

class ObjectTracker:
    def __init__(self, speechm):
        self.objects = []
        self.id_increment = 0
        self.speechm = speechm
        # todo account for offscreen

    def update(self, contours, recognized_faces):
        self.update_object_movements(contours)
        self.update_object_recognition(recognized_faces)

    # Updating object movements is performed in two steps.
    # 1. For each new position, update an existing one, or create a new. 
    # 2. Remove unupdated existing ones.
    # TODO: eg. frame 1: two objects, frame 2: one obj, between orig two. how to handle?
    def update_object_movements(self, new_contours):
        for obj in self.objects:
            obj.updated = False

        for new_contour in new_contours:
            new_centroid = self.get_centroid(new_contour)

            # find dist and index to closest existing centroid
            dists = [self.sqr_dist(new_centroid, obj.centroid) for obj in self.objects]
            if len(dists) > 0:
                j, dist = min((jd for jd in enumerate(dists)), key=lambda x:x[1])
                matched = True if dist < 100**2 else False # only count if movement plausible
            else:
                matched = False

            # if there's a match, update, if not create new object.
            if matched:
                # import ipdb; ipdb.set_trace()
                self.objects[j].centroid = new_centroid
                self.objects[j].contour = new_contour
                self.objects[j].updated = True
            else: # new object
                print "Identified new object %d" % self.id_increment
                self.objects.append(
                    Object(
                        contour=new_contour,
                        centroid=new_centroid,
                        id=self.id_increment,
                        updated=True
                    )
                )
                self.id_increment += 1
                
        # Keep only updated objects
        for x in self.objects:
            if not x.updated and x.name is not None:
                self.speechm.say_goodbye(x.name)
        self.objects = [x for x in self.objects if x.updated]


    def update_object_recognition(self, recognized_faces):
        for (face, label, unconfidence) in recognized_faces:
            objs = [(i, obj) for (i, obj) in enumerate(self.objects) if 
                    self.rect_mostly_inside_polygon(face, obj.contour)]

            if len(objs) == 1: # only 1 plausible match
                (i, obj) = objs[0]
                if self.objects[i].name != label:
                    self.objects[i].name = label
                    self.speechm.say_hello(label)
            else:
                print "two plausible body matches for %s." % label


    def get_centroid(self, cnt):
        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        return (centroid_x, centroid_y)

    def rect_mostly_inside_polygon(self, rect, cnt):
        # Mostly inside if center of rect is inside polygon
        (x, y, w, h) = rect
        pt = (x+w/2, y+h/2)
        return cv2.pointPolygonTest(cnt, pt, False) == 1
        # 1 is inside, -1 is outside, 0 is on

    def sqr_dist(self, p1, p2):
        return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def main():
    cap = cv2.VideoCapture(0)
    faceDetector = FaceDetector()
    objDetector = MovingObjectDetector()
    faceRecognizer = FaceRecognizer()
    faceRecognizer.load_newest_model()
    objDetector.start_calibration()
    bodyDetector = BodyDetector()
    speechm = SpeechModule()
    speechm.start()
    objTracker = ObjectTracker(speechm)
    # create object tracker use it

    frame_counter = 0
    while ( cap.isOpened() ):
        ret, frame = cap.read()

        contours = objDetector.detect_moving_objects(frame)

        if frame_counter % 5 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = faceDetector.detect_faces(gray)
            recognized_faces = faceRecognizer.recognize(gray, faces)
            bodies = bodyDetector.detect_bodies(frame)

        objTracker.update(contours, recognized_faces)

        # Draw tracked objects
        for obj in objTracker.objects:
            if obj.name:
                cv2.drawContours(frame,[obj.contour],0,(250,120,120),2)
                cv2.putText(frame, 
                    obj.name,
                    org = (max(obj.centroid[0]-20,0), max(obj.centroid[1]-10,0)), 
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1.0,
                    color = (250,120,120),
                    thickness = 2
                )
            else:
                cv2.drawContours(frame,[obj.contour],0,(0,255,0),2)
                cv2.putText(frame, 
                    "unknown %d" % obj.id,
                    org = (max(obj.centroid[0]-20,0), max(obj.centroid[1]-10,0)), 
                    fontFace = cv2.FONT_HERSHEY_PLAIN,
                    fontScale = 1.0,
                    color = (250,120,120),
                    thickness = 2
                )

        # Draw around face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Draw face recognition annotations
        for (face, label, unconfidence) in recognized_faces:
            if unconfidence >= 50:
                text = "%s ? (var: %i)" % (label, unconfidence)
            else:
                text = "%s (var: %i)" % (label, unconfidence)

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

        # Draw bodies
        for (x, y, w, h) in bodies:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 122, 255), 2)
        
        cv2.imshow('img', frame)
        cv2.moveWindow('img', 700, 0)

        k = cv2.waitKey(10) & 0xff
        if k == 27:
            faceRecognizer.save()
            break
        if k == ord('c'):
            objDetector.start_calibration()
        if k == ord('1'):
            faceRecognizer.start_training("Rui Lin")
        if k == ord('2'):
            faceRecognizer.start_training("Jessica")
        if k == ord('3'):
            faceRecognizer.start_training("a")
        if k == ord('4'):
            faceRecognizer.start_training("a")
        if k == ord('s'): # save new snapshot
            faceRecognizer.save(newfile=True)

        frame_counter = (frame_counter + 1) % 10

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()