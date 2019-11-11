#!/usr/bin/env python3

import os
import sys

import time
import cv2
import numpy as np
import sdl2
import sdl2.ext

from display import Display

W = 1920//2
H = 1080//2

disp = Display(W,H)

#orb = cv2.ORB_create()

class FeatureExtractor():
    GX = 16//2
    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(1000)

    def extract(self, img):
        # run detect in grid
        sx = img.shape[1]//self.GX
        sy = img.shape[0]//self.GY
        akp = []
        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                img_chunk = img[ry:ry+sy, rx:rx+sx]
                kp = self.orb.detect(img_chunk, None)
                for p in kp:
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    akp.append(p)
        return akp


def process_frame(img):
    img = cv2.resize(img, (W,H))
    #kp1, des1 = orb.detectAndCompute(img, None)
    kp = fe.extract(img)

    for p in kp:
        u,v = map(lambda x: int(round(x)), p.pt)
        cv2.circle(img, (u,v), color=(0,255,0), radius = 3)

    disp.paint(img)

fe = FeatureExtractor()

if __name__ == "__main__":
    cap = cv2.VideoCapture("car_dash.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
