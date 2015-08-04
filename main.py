import cv2
import numpy as np


cap = cv2.VideoCapture(0)

while ( cap.isOpened() ):
    ret, img = cap.read()
    debug_drawing = np.zeros(img.shape,np.uint8)
    cv2.imshow('debug',debug_drawing)
    cv2.imshow('img',img)

    k = cv2.waitKey(10)
    if k == 27:
        break

cv2.destroyAllWindows()