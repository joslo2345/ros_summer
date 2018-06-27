import cv2;
import numpy as np;

'''
def get_contour_precedence(contour, cols):
    tolerance_factor = 10
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]
'''

# Read image
im_in = cv2.imread("carretera.jpg", cv2.IMREAD_GRAYSCALE)

blurred_image_1 = cv2.GaussianBlur(im_in,(13,13),0)
cv2.imshow("GaussianBlur",blurred_image_1)
cv2.waitKey(0)
blurred_image_2 = cv2.bilateralFilter(im_in,9,75,75)
cv2.imshow("bilateralFilter",blurred_image_2)
cv2.waitKey(0)

#threshold
#th, im_th = cv2.threshold(blurred_image_2, 126, 255, cv2.THRESH_OTSU)
th, im_th = cv2.threshold(blurred_image_2, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow("Thresholded Image", im_th)
cv2.waitKey(0)

#contours

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
dilation = cv2.dilate(im_th,kernel,iterations = 1)
cv2.imshow("dilate",dilation)
cv2.waitKey(0)

kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(im_th,kernel,iterations = 1)
cv2.imshow("erosion",erosion)
cv2.waitKey(0)

#bakcground substractor
kernel = cv2.getStructuringElement(cv2.MORPH_CLOSE,(11,11))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
fgmask = fgbg.apply(im_in)
cv2.imshow("bakcground substractor",fgmask)
cv2.waitKey(0)

#im2, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
im2, contours, hierarchy = cv2.findContours(im_th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#for x in contours[0]:
#    print(x)
'''
contours.sort(key=lambda x:get_contour_precedence(x, im_in.shape[1]))

# For debugging purposes.
for i in xrange(len(contours)):
    img = cv2.putText(im_in, str(i), cv2.boundingRect(contours[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [0,0,125])
    cv2.imshow("show",img)
    cv2.waitKey(50)
'''

ordered_contours = sorted(contours,key = cv2.contourArea, reverse = True)
cv2.drawContours(im_in, ordered_contours, -1, (0,255,0), 3)
cv2.imshow("contours",im_in)
cv2.waitKey(0)

#for x in ordered_contours:
    #im_con = cv2.drawContours(im_in, x, -1, (0,255,0), 3)
    #cv2.imshow("contours",im_con)
    #cv2.waitKey(0)
    #print(x)
    #raw_input("Press Enter to continue...")
