import cv2
import Image
import numpy as np

import face_alignment_ext

class FaceAligner:
  def __init__(self):
    eye_path = "data/haarcascade_eye.xml"
    self.eye_cascade = cv2.CascadeClassifier(eye_path)

  def align_face(self, img):
    img = cv2.equalizeHist(img) # increase contrast
    img = cv2.resize(img, (200,200)) # todo allow it to detect eyes more!
    eyes = self.eye_cascade.detectMultiScale(
        img,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(40, 40),
        flags=0 #cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    for (x,y,w,h) in eyes:
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 122, 255), 2)
    cv2.imshow('test', img)

    # exactly 2 distinct eyes found
    if len(eyes) != 2 or self.overlap(eyes[0], eyes[1], 0.3):
      return None

    eyes = sorted(eyes, key=lambda x:x[0]) # sorts by first element of tuple, the x-dir
    left_eye = self.get_center_of_rect(eyes[0])
    right_eye = self.get_center_of_rect(eyes[1])

    aligned_face = face_alignment_ext.CropFace(img, left_eye, right_eye, 
      offset_pct=(0.2,0.2), dest_sz = (100,100))

    cv2.imshow('aligned_face', aligned_face)

    return aligned_face

  def get_center_of_rect(self, rect):
    x, y, w, h = rect
    return (x+w/2, y+h/2)

  # Do rectangles overlap, given % allowance of overlap to not count.
  def overlap(self, rectA, rectB, allowance):
    ax, ay, aw, ah = rectA
    bx, by, bw, bh = rectB
    pct = 1 - allowance

    if ax > bx + bw*pct or bx > ax + aw*pct or ay > by + bh*pct or by > ay + ah*pct:
      return False
    return True



