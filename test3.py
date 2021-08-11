import cv2
import numpy as np
from scipy import ndimage


a = np.array(([0,1,1,1,1], [0,0,1,1,1], [0,1,1,1,1], [0,1,1,1,0],[0,1,1,0,0]))

dist = ndimage.distance_transform_edt(a)

print(dist)