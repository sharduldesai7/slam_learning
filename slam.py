#!/usr/bin/env python3

import os
import sys

import time
import cv2
import numpy as np
import sdl2
import sdl2.ext

from display import Display
from extractor import Extractor

W = 1920//2
H = 1080//2
F = 1       #camera focal length

disp = Display(W,H)
K = np.array([[F,0,W//2],[0,F,H//2],[0, 0, 1]])     #camera intrinsic matrix


#orb = cv2.ORB_create()



fe = Extractor(K)

def process_frame(img):
    img = cv2.resize(img, (W,H))
    #kp1, des1 = orb.detectAndCompute(img, None)
    matches = fe.extract(img)

    if matches is None:
        return

    #print ("len of matches: %s" % len(matches))
    for pt1, pt2 in matches:
        #print (pt1, pt2)
        #u1,v1 = map(lambda x: int(round(x)), pt1)
        #u2,v2 = map(lambda x: int(round(x)), pt2)

        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)
        cv2.circle(img, (u1,v1), color=(0,255,0), radius = 3)
        cv2.line(img, (u1,v1), (u2,v2), color=(255,0,0))

    disp.paint(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("car_dash.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
