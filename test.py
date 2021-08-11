import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from blend import *
#from sklearn.preprocessing import normalize

QUERY = 1
TRAIN = 2
MIN_MATCH_COUNT = 4 # minimum 4

img1_color = cv.imread('./data/test1.jpg') # queryImage
img2_color = cv.imread('./data/test2.jpg') # trainImage (Reference)

img1 = cv.cvtColor(img1_color, cv.COLOR_BGR2GRAY) # queryImage
img2 = cv.cvtColor(img2_color, cv.COLOR_BGR2GRAY) # trainImage (Reference)

# Initiate KAZE detector
kaze = cv.KAZE_create()

# find the keypoints and descriptors with KAZE
kp1, des1 = kaze.detectAndCompute(img1,None)
kp2, des2 = kaze.detectAndCompute(img2,None)

img1_kaze = cv.drawKeypoints(img1_color, kp1, None, (255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kaze = cv.drawKeypoints(img2_color, kp2, None, (255,0,0),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# FLANN
FLANN_INDEX_KDTREE = 1 #KD tree is 1
index_params = dict(algorithm = 4, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, 2)
# print(matches)

# Brute force matching
# print(matches[100][0].queryIdx, matches[100][0].queryIdx)
# for i in range(len(matches)):
    #sort each array based on distance

# bf = cv.BFMatcher()
# matches = bf.radiusMatch(des1,des2,16)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)


if len(matches)>MIN_MATCH_COUNT: # DEBUG, orginal good
    src_pts =  np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts =  np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape #,d
    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    # dst = cv.perspectiveTransform(pts,M)
    # img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

for i in range (len(src_pts)):
    print(src_pts[i], dst_pts[i])

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params) # good

# transformed_img = cv.warpPerspective(img1_color,
#                     M, (w, h))

# cv.imshow("test", helperBlendImages( transformed_img,img2_color ))
# cv.waitKey()

cv.imshow('gray', img3)
cv.waitKey()