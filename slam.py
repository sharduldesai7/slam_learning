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

disp = Display(W,H)

#orb = cv2.ORB_create()



fe = Extractor()

def process_frame(img):
    img = cv2.resize(img, (W,H))
    #kp1, des1 = orb.detectAndCompute(img, None)
    matches = fe.extract(img)

    if matches is None:
        return

    for p in kps:
        #u,v = map(lambda x: int(round(x)), p.pt)
        u,v = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (u,v), color=(0,255,0), radius = 3)

    disp.paint(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("car_dash.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
