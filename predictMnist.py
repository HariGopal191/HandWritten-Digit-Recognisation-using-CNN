# Importing the Keras libraries and packages
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.models import load_model
model = load_model('mnist_keras_cnn_model.h5')
from PIL import Image
import numpy as np
import cv2


def x_cord_contour(contour):
    # This function take a contour from findContours
    # it then outputs the x centroid coordinates

    if cv2.contourArea(contour) > 10:
        M = cv2.moments(contour)
        return (int(M['m10'] / M['m00']))
    else:
        return -1

def makeSquare(not_square):
    # This functions taken an image and makes the different square
    # It adds black pixels as the padding where needed

    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Height = ", height, "Width = ", width)
    if(height == width):
        square = not_square
        return square
    else:
        doublesize = cv2.resize(not_square, (2*width, 2*height), interpolation = cv2.INTER_CUBIC)
        height = height * 2
        width = width * 2
        # print("New Height = ", height, width = ", width)
        if(height > width):
            pad = (height - width) // 2
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=BLACK)
        else:
            pad = ( width - height ) // 2
            # print("Padding = ", pad)
            doublesize_square = cv2.copyMakeBorder(doublesize, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value = BLACK)
    doublesize_square_dim = doublesize_square.shape
    # print("Sq Height = ", doublesize_square_dim[0], "sq Width = ", doublesize_square_dim[1])
    return doublesize_square

def verifyOverlap(l1, r1, l2, r2):
    # If one rectangle is on left side of other
    if(l1[0] > r2[0] or l2[0] > r1[0]):
        return False

    # If one rectangle is above other
    if(l1[1] < r2[1] or l2[1] < r1[1]):
        return False

    return True

def get_contour_precedence(contour, cols):
    tolerance_factor = 100
    origin = cv2.boundingRect(contour)
    return ((origin[1] // tolerance_factor) * tolerance_factor) * cols + origin[0]



#-----------------------------------------------------------------------------------------



image = cv2.imread('./input/004.jpeg')
#image = cv2.imread('/mnt/e/miniproject/myPro/input/test_img.png')
#image = cv2.imread('E:/miniproject/myPro/input/numbers.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#print(gray.shape)

if(gray.shape == (28, 28)):
    y_pred = model.predict(gray.reshape(1, 28, 28, 1))
    number = str(int(float(np.where(y_pred == np.amax(y_pred))[1][0])))
    print(number)
    __import__('sys').exit(1)


ret, gray1 = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY_INV)
#gray1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
#cv2.imshow("Threshhold", gray1)
# Blur image then find edges using Canny

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

blurred1 = cv2.GaussianBlur(gray1, (5, 5), 0)
blur1 = cv2.GaussianBlur(gray1, (5, 5), 0)

edged = cv2.Canny(blurred1, 30, 150)
#cv2.imshow("Edges", edged)

# Find Contours
contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(hierarchy)
# Sort out contours left to right by using thier x coordinate
#contours = sorted(contours, key = cv2.contourArea, reverse = True)
#contours = sorted(contours, key =lambda x:get_contour_precedence(x, blur1.shape[1]), reverse = False)
contours = sorted(contours, key = x_cord_contour, reverse = False)

w1 = [cv2.boundingRect(c)[2] for c in contours]
w_avg = (( sum(w1)/len(w1) ) + (max(w1)-min(w1))/2 ) /1.5
h1 = [cv2.boundingRect(c)[3] for c in contours]
h_avg = (( sum(h1)/len(h1) ) + (max(h1)-min(h1))/2 ) /1.5

# Create empty array to store entire number
full_number = []
elements = []

# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)
    #print(elements)

    if(not elements):
        elements.append((x, y, x+w, y+h))
    elif(any([1 for i in elements if(verifyOverlap( (i[0], i[1]), (i[2], i[3]), (x, y), (x+w, y+h) ))])):
        continue
    else:
        elements.append((x, y, x+w, y+h))

    if(w >=w_avg and h >=h_avg):
        roi = blurred1[y:y+h, x:x+w]

        #ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.rectangle(blur1, (x, y), (x+w, y+h), (255, 255, 255), 2)
		#blur1 = cv2.putText(blur1, number, cv2.boundingRect(contours[i])[:2], cv2.FONT_HERSHEY_COMPLEX, 1, [125])


        #roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        while (True):
            cv2.imshow("Image", roi)
            k = cv2.waitKey(50) & 0xFF
            if k == ord('q'):
                break

        squared = makeSquare(roi)
        #print(squared.shape)
        final = cv2.resize(squared, (28, 28), interpolation = cv2.INTER_AREA)
        #print(final.shape)
        y_pred = model.predict(final.reshape(1, 28, 28, 1))
        print(y_pred)
        number = str(int(float(np.where(y_pred == np.amax(y_pred))[1][0])))
        print(number)
        full_number.append(number)
    else:
        roi = blurred1[y:y+h, x:x+w]

        #ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)

        #roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

        squared = makeSquare(roi)
        #print(squared.shape)
        final = cv2.resize(squared, (28, 28), interpolation = cv2.INTER_AREA)
        #print(final.shape)
        y_pred = model.predict(final.reshape(1, 28, 28, 1))
        number = str(int(float(np.where(y_pred == np.amax(y_pred))[1][0])))
        #print(number)
        if(number==7 or number ==1):
            cv2.rectangle(blur, (x, y), (x+w, y+h), (255, 255, 255), 2)
            print(y_pred)
            while (True):
                cv2.imshow("Image_ding_dong", roi)
                k = cv2.waitKey(50) & 0xFF
                if k == ord('q'):
                    break

            full_number.append(number)



print("The number is : " + ''.join(full_number))

while (True):
    cv2.imshow("Image", blur1)
    k = cv2.waitKey(50) & 0xFF
    if k == ord('q'):
        break

cv2.destroyAllWindows()

