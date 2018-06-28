import cv2;
import numpy as np;

# Read image
image = cv2.imread("carretera_4.jpg")
im_in = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# cropping image
#third_x = int(im_in)
third_y = int(im_in.shape[0]/2)
width = int(im_in.shape[1])
height = int(im_in.shape[0])
cropped_image = im_in[third_y:height,0:width]
cv2.imshow("cropped image",cropped_image)
cv2.waitKey(0)

# Gaussian Blur
blurred_image_1 = cv2.GaussianBlur(cropped_image,(13,13),0)
#cv2.imshow("GaussianBlur",blurred_image_1)
#cv2.waitKey(0)

# bilateral Filter

blurred_image_2 = cv2.bilateralFilter(cropped_image,15,55,55)
#cv2.imshow("bilateralFilter",blurred_image_2)
#cv2.waitKey(0)

#threshold

th, im_th = cv2.threshold(blurred_image_1, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Thresholded Image", im_th)
cv2.waitKey(0)

#Morphological transformantion

# closing
kernel = np.ones((9,9),np.uint8)
closing = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel, iterations = 1)
cv2.imshow("gradient",closing)
cv2.waitKey(0)


# find contours

im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#for x in contours[0]:
#    print(x)

#ordered contours

ordered_contours = sorted(contours,key = cv2.contourArea, reverse = True)
#cv2.drawContours(cropped_image, ordered_contours, -1, (0,255,0), 3)
#cv2.imshow("contours",cropped_image)
#cv2.waitKey(0)

for x in ordered_contours:
    #print(cv2.contourArea(x))
    if cv2.contourArea(x) > 50:
        print(cv2.contourArea(x))
        M = cv2.moments(x)
        cv2.drawContours(cropped_image, x, -1, (0,255,0), 3)
        x,y,w,h = cv2.boundingRect(x)
        cv2.rectangle(cropped_image,(x,y),(x+w,y+h),(0,255,0),2)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.circle(cropped_image,(cx,cy),1,(0,255,0),3)
        cv2.imshow("contours",cropped_image)
        cv2.waitKey(500)
cv2.imshow("lines",cropped_image)
cv2.waitKey(0)
