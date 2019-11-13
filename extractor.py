import cv2
import numpy as np

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
from skimage.transform import EssentialMatrixTransform

#convert [[x,y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis = 1)

class Extractor():
#    GX = 16//2
#    GY = 12//2

    def __init__(self, K):
        self.orb = cv2.ORB_create() #ORB creator
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)       #Brute force matcher
        self.last = None
        self.K = K
        self.Kinv = np.linalg.inv(self.K)


    def denormalize(self, pt):
        ret = np.dot(self.K,np.array([pt[0], pt[1], 1.0]))

        return int(round(ret[0])), int(round(ret[1]))

    def extract(self, img):         
        #goodFeaturesToTrack is an opencv function to extract keypoints

        #detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel = 0.01, minDistance = 1)

        #extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)        #kps = keypoints, des = descriptors

        #matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k = 2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))
        
        #filter: using fundamental filter - governs how points correspond to each other
        if len(ret) > 0:
            ret = np.array(ret)

            ret[:, 0, :] = np.dot(self.Kinv, add_ones(ret[:, 0, :]).T).T[:, 0:2]
            ret[:, 1, :] = np.dot(self.Kinv, add_ones(ret[:, 1, :]).T).T[:, 0:2]

            model, inliers = ransac((ret[:, 0], ret[:, 1]),
                                    FundamentalMatrixTransform,
                                    min_samples = 8,
                                    residual_threshold = 1,
                                    max_trials = 100)

            ret = ret[inliers]

        #return
        self.last = {'kps' : kps, 'des' : des}
        return ret

