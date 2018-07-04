import cv2;
import numpy as np;
import matplotlib.pyplot as plt
import operator

# Read image
image = cv2.imread("white_test_2.jpg")

# cropping image
#third_x = int(im_in)
third_y = int(image.shape[0]/2)
width = int(image.shape[1])
height = int(image.shape[0])
print(third_y)
print(width)




cropped_image = image[third_y:height,0:width]

def select_rgb_white_yellow(image):
    # white color mask
    #lower = np.uint8([220, 220, 20])
    #upper = np.uint8([255, 255, 255])
    lower = np.uint8([200, 200, 200])
    upper = np.uint8([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked
cv2.imshow("masked",select_rgb_white_yellow(cropped_image))
cv2.waitKey(0)
