import cv2
import numpy as np
from blend import *

# Open the image files.
img1_color = cv2.imread("./data/test7.jpg")  # Image to be aligned.
img2_color = cv2.imread("./data/test8.jpg")    # Reference image.
  
# Convert to grayscale.
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY)
height, width = img2.shape

KAZE = cv2.KAZE_create()

kp1, d1 = KAZE.detectAndCompute(img1, None)
kp2, d2 = KAZE.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=150)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(d1,d2,k=2)

ratio_thresh = 0.85
good_matches = []
for m,n in matches:
    if m.distance < ratio_thresh * n.distance:
        good_matches.append(m)

# homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC, RANSAC_REPROJ_THRESHOLD)

img_matches = np.empty((max(img1.shape[0], img2.shape[0]), img1.shape[1]+img2.shape[1], 3), dtype=np.uint8)
cv2.drawMatches(img1_color, kp1, img2_color, kp2, good_matches, img_matches, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
#-- Show detected matches
cv2.imshow('Good Matches', img_matches)
cv2.waitKey()
