import cv2;
import numpy as np;
import matplotlib.pyplot as plt
import operator
# Read image
image = cv2.imread("carretera_4.jpg")
im_in = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# cropping image
#third_x = int(im_in)
third_y = int(im_in.shape[0]/2)
width = int(im_in.shape[1])
print "width/2 = ", width/2 , "width = ",  width
height = int(im_in.shape[0])

cropped_image = im_in[third_y:height,0:width]
cropped_image_2 = image[third_y:height,0:width]
#cv2.imshow("cropped image",cropped_image)
#cv2.waitKey(0)

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
#cv2.imshow("Thresholded Image", im_th)
#cv2.waitKey(0)

#Morphological transformantion

# closing
kernel = np.ones((9,9),np.uint8)
closing = cv2.morphologyEx(im_th, cv2.MORPH_CLOSE, kernel, iterations = 1)
#cv2.imshow("gradient",closing)
#cv2.waitKey(0)


# find contours

im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
#for x in contours[0]:
#    print(x)

#ordered contours

ordered_contours = sorted(contours,key = cv2.contourArea, reverse = True)
#cv2.drawContours(cropped_image, ordered_contours, -1, (0,255,0), 3)
#cv2.imshow("contours",cropped_image)
#cv2.waitKey(0)
cx_array = []
cy_array = []
for x in ordered_contours:
    #print(cv2.contourArea(x))
    if cv2.contourArea(x) > 50:
        # print area
        #print(cv2.contourArea(x))
        M = cv2.moments(x)
        #cv2.drawContours(cropped_image_2, x, -1, (0,0,255), 3)
        #x,y,w,h = cv2.boundingRect(x)
        #cv2.rectangle(cropped_image,(x,y),(x+w,y+h),(0,255,0),2)
        cx = int(M['m10']/M['m00'])
        cx_array.append(cx)
        cy = int(M['m01']/M['m00'])
        cy_array.append(cy)
        #cv2.circle(cropped_image_2,(cx,cy),10,(0,255,0),3)
        #cv2.imshow("contours",cropped_image_2)
        #cv2.waitKey(500)

# arrays of centroids xs and ys values
#print(cx_array)
cx_array_sorted = sorted(cx_array,reverse=True)
#print(cx_array_sorted)
#print(cy_array)

#ploting centroids of the blobs
cy_array_invert = [-x for x in cy_array]
plt.plot(cx_array,cy_array_invert,"bo")

# parameter to exclude some blobs
center_of_way = cx_array_sorted[0]-int(width/2)
half_center_of_way = int(center_of_way/2)
range_excluding = int(width/2)-half_center_of_way

# excluding blobs
cx_include = []
cy_include = []
for x in ordered_contours:
    #print(cv2.contourArea(x))
    if cv2.contourArea(x) > 50:
        M = cv2.moments(x)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if cx >= range_excluding:
            cx_include.append(int(cx))
            cy_include.append(int(cy))
            cv2.drawContours(cropped_image_2, x, -1, (0,0,255), 3)
            cv2.circle(cropped_image_2,(cx,cy),10,(0,255,0),3)
            #cv2.imshow("contours",cropped_image_2)
            #cv2.waitKey(500)

# centroids included
print(cx_include)
print(cy_include)
print "\n"

list_centroids = {}
for x in range(len(cx_include)):
    list_centroids[cx_include[x]] = cy_include[x]
print(list_centroids)
list_centroids_sorted = sorted(list_centroids.items(), key=operator.itemgetter(0),reverse=True)
#sorting centroids
print(list_centroids_sorted)
print "\n"
# preparing centroids for linear regresion
x = []
y = []
for i in list_centroids_sorted:
    x.append(i[0])
    y.append(i[1])
print(x)
print(y)
print "\n"

# exclude most right centroid
x = x[1:]
y = y[1:]
print(x)
print(y)

# linear regresion procedure
x = np.array(x, dtype=np.float64)
y = np.array(y, dtype=np.float64)
n_elements = len(x)
x_sum = sum(x)
x2_sum = sum(x**2)
y_sum = sum(y)
xy_sum = sum(x*y)

slope = ((n_elements*xy_sum)-(x_sum*y_sum))/((n_elements*x2_sum)-(x_sum**2))
b = (y_sum/n_elements)-slope*(x_sum/n_elements)
print "y = %s*x + %s" %(slope,b)
#x_test = list(range(range_excluding,(int(width/2)+range_excluding),10))
x_test = [0,int(width/2)+range_excluding]
x_test = np.array(x_test, dtype=np.float64)

y_calculated = slope*x_test + b
plt.plot(x_test,-1*y_calculated)
plt.savefig("grafica.png")

cv2.line(cropped_image_2,
(int(x_test[0]),int(y_calculated[0])),
(int(x_test[1]),int(y_calculated[1])),
(255,0,0),
thickness=4)

# final image with blobs and centroids
cv2.imshow("lines",cropped_image_2)
cv2.waitKey(0)
