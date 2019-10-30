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
    #This function take a contour from findContours
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

'''
image = cv2.imread('digits.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)
# Split the image to 5000 cells, 20x20 size
# This gives us a 4-dim array: 50 x 100 x 20 x 20
cells = [ np.hsplit(row, 100) for row in np.vsplit(gray, 50) ]
# Convert the List datya type to Numpy Array of shape (50, 100, 20, 20)
x = np.array(cells)
pred = x[:,:1]
for img in pred:
	#img = img.resize((28,28))
	#im2arr = np.array(img)
	#im2arr = im2arr.reshape(1,28,28,1)
	# Predicting the Test set results
	img = img.reshape(20, 20)
	#print(img.shape)
	img = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
	#print(img.shape)
	y_pred = model.predict(img.reshape(1, 28, 28, 1))
	print(y_pred)
'''

image = cv2.imread('/mnt/e/miniproject/myPro/input/numbers.jpg')
#image = cv2.imread('/mnt/e/miniproject/myPro/input/test_img.png')
#image = cv2.imread('E:/miniproject/myPro/input/numbers.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#print(gray.shape)

if(gray.shape == (28, 28)):
    y_pred = model.predict(gray.reshape(1, 28, 28, 1))
    number = str(int(float(np.where(y_pred == np.amax(y_pred))[1][0])))
    print(number)
    __import__('sys').exit(1)

# Blur image then find edges using Canny
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edged = cv2.Canny(blurred, 30, 150)

# Find Contours
contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Sort out contours left to right by using thier x coordinate
contours = sorted(contours, key = x_cord_contour, reverse = False)

# Create empty array to store entire number
full_number = []

# loop over the contours
for c in contours:
    # compute the bounding box for the rectangle
    (x, y, w, h) = cv2.boundingRect(c)

    if(w >= 5 and h >=25):
        roi = blurred[y:y+h, x:x+w]
        
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        #roi = cv2.adaptiveThreshold(roi,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        '''
        while (True):
            cv2.imshow("Image", roi)
            k = cv2.waitKey(50) & 0xFF
            if k == ord('q'):
                break
        '''
        squared = makeSquare(roi)
        #print(squared.shape)
        final = cv2.resize(squared, (28, 28), interpolation = cv2.INTER_AREA)
        #print(final.shape)
        y_pred = model.predict(final.reshape(1, 28, 28, 1))        
        number = str(int(float(np.where(y_pred == np.amax(y_pred))[1][0])))
        full_number.append(number)
        

print("The number is : " + ''.join(full_number))
