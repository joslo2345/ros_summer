import cv2;
import numpy as np
import matplotlib.pyplot as plt
import operator

# Read image
image = cv2.imread("prueba_1.jpg")
im_in = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
im_inverted = cv2.bitwise_not(im_in)
#cv2.imshow("inverted",im_inverted)
#cv2.waitKey(0)

# white color mask
lower = np.uint8([200, 200, 200])
upper = np.uint8([255, 255, 255])
white_mask = cv2.inRange(image, lower, upper)
# yellow color mask
lower = np.uint8([190, 190,   0])
upper = np.uint8([255, 255, 255])
yellow_mask = cv2.inRange(image, lower, upper)
# combine the mask
mask = cv2.bitwise_or(white_mask, yellow_mask)
masked = cv2.bitwise_and(image, image, mask = mask)

cv2.imshow("carretera",masked)
cv2.waitKey(0)
