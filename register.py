import cv2
import os
import numpy as np

def helperRegisterImages(images, radius):

    matchThreshold = 10
    maxRatio       = 0.6
    confidence     = 99.9
    maxDistance    = 2
    maxNumTrials   = 2000
    numImages      = len(images.keys())


    features   = {}
    points     = {}
    grayImages = {}
    tforms     = {}

    for i in range(1, numImages + 1):
        grayImages[i] = cv2.cvtColor(images[i], cv2.COLOR_RGB2GRAY)
        KAZE = cv2.KAZE_create()
        points[i], features[i] = KAZE.detectAndCompute(grayImages[i], None)

    for ii in range(2,numImages + 1):

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=150)   # or pass empty dictionary
        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(features[ii-1],features[ii],k=2)
  
        matchesMask = [[0,0] for i in range(len(matches))]
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.7*n.distance:
                matchesMask[i]=[1,0]

        draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)

        img3 = cv2.drawMatchesKnn(grayImages[ii-1],points[ii-1],grayImages[ii],points[ii],matches,None,**draw_params)
        cv2.imshow("test", img3)
        cv2.waitKey(0)

    return 0
