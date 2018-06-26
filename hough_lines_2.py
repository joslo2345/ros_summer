import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("carretera.jpg")
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur
blurred_image = cv2.GaussianBlur(image_gray,(9,9),0)

# Canny edge detection
edges_image = cv2.Canny(blurred_image,50,120)


# Hough lines
rho = 1
theta = np.pi/180
threshold = 155

hough_lines = cv2.HoughLines(edges_image, rho, theta,threshold)
hough_lines_image = np.zeros_like(image)

for line in hough_lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(hough_lines_image,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.imshow("line",hough_lines_image)

image_weighted = cv2.addWeighted(hough_lines_image,0.8,image,1.0,0)
'''
plt.figure(figsize = (30,20))
plt.subplot(131)
plt.imshow(image)
plt.subplot(132)
plt.imshow(edges_image, cmap='gray')
plt.subplot(133)
plt.imshow(image_weighted, cmap='gray')
plt.show()
'''

#cv2.imshow("lines",image_weighted)
#cv2.waitKey()
