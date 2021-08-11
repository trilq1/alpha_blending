import numpy as np
import cv2 as cv

from blend import *

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
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = 4, trees = 5)
search_params = dict(checks = 50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, 2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

src_pts =  np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts =  np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

_, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
matchesMask = mask.ravel().tolist()

print(f"{len(src_pts)} match pairs:\n")
for i in range (len(src_pts)):
    print(src_pts[i], dst_pts[i])

print("\n===================\n")

error = 1 / len(src_pts) * np.sum( np.sqrt( np.sum( np.sum( ( dst_pts - src_pts ) ** 2, axis = 1), axis = 1 ) ) )

print("Average pixel distance:  ", error)

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

cv.imshow('gray', img3)
cv.waitKey()