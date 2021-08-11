import cv2
import os
import numpy as np
from blend import *
from register import *
from stitch import *

from scipy import ndimage


def main():
    bird_view_dir = "./data"

    files = sorted([i for i in os.listdir(bird_view_dir) if ".jpg" in i])

    numCameras = 8
    leftRadius = 15
    rightRadius = 10
    allRadius = 60

    leftImgs = {}
    rightImgs = {}

    i = 1
    for f in files:
        if i <= 4:
            leftImgs[i] = cv2.imread(os.getcwd()+"/data/" + f, cv2.IMREAD_COLOR)
        else:
            rightImgs[i-4] = cv2.imread(os.getcwd()+"/data/" + f, cv2.IMREAD_COLOR)
        i += 1
    
    # Combine the first four images
    tforms = helperRegisterImages(rightImgs, leftRadius)
    
    leftSideView, Rleft = helperStitchImages(leftImgs, tforms)


    # Combine the last four images

    return 0



if __name__ == "__main__":
    main()
