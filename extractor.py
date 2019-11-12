import cv2
import numpy as np

class Extractor():
#    GX = 16//2
#    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(100) #ORB creator
        self.bf = cv2.BFMatcher()       #Brute force matcher
        self.last = None

    def extract(self, img):         
        '''
        # run detect in grid
        #reduce the image size using GX and GY. Compute features for them and return list with features
        sx = img.shape[1]//self.GX
        sy = img.shape[0]//self.GY
        akp = []
        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                img_chunk = img[ry:ry+sy, rx:rx+sx]
                kp = self.orb.detect(img_chunk, None)   #finds good keypoints
                for p in kp:
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    akp.append(p)
        
        return akp
        '''
        #goodFeaturesToTrack is an opencv function to extract keypoints
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 3)
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)        #kps = keypoints, des = descriptors
   
        matches = None
        if self.last is not None:
            matches = self.bf.match(des, self.last['des'])
            matches =  zip([kps[m.queryIdx] for m in matches], [kps[m.trainIdx] for m in matches])
            #print (matches)

        self.last = {'kps' : kps, 'des' : des}
        return matches

