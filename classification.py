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
        square == not_square
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

def resize_to_pixel(dimensions, image):
    # This function then re-size an image to the specified dimensions

    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized  = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if(height_r > width_r):
        resized = cv2.copyMakeBorder(resized, 0, 0, 0, 1, cv2.BORDER_CONSTANT, value = BLACK)
    if(height_r < width_r):
        resized = cv2.copyMakeBorder(resized, 1, 0, 0, 0, cv2.BORDER_CONSTANT, value = BLACK)
    p = 2
    ReSizedImg = cv2.copyMakeBorder(resized, p, p, p, p, cv2.BORDER_CONSTANT, value = BLACK)
    img_dim = ReSizedImg.shape
    height = img_dim[0]
    width = img_dim[1]
    # print("Padding Height = ", height, "Width = ", width)
    return ReSizedImg


# Let's take a look at our digits dataset
image = cv2.imread('OpenCV-with-Python-master/images/digits.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
small = cv2.pyrDown(image)

# Split the image to 5000 cells, 20x20 size
# This gives us a 4-dim array: 50 x 100 x 20 x 20
cells = [ np.hsplit(row, 100) for row in np.vsplit(gray, 50) ]

# Convert the List datya type to Numpy Array of shape (50, 100, 20, 20)
x = np.array(cells)
print("The shape of our cells array: " + str(x.shape))

# Split the full data set into two segments
# One will be used for Training the Model, the other as a Test data set
train = x[:,:100].reshape(-1, 400).astype(np.float32) # Size = (3500, 400)
#test = x[:,98:100].reshape(-1, 400).astype(np.float32) # Size = (1500, 400)

# Create labels for Train and Test data set
k = [0,1,2,3,4,5,6,7,8,9]
train_labels = np.repeat(k, 500)[:, np.newaxis]
#test_labels = np.repeat(k, 10)[:, np.newaxis]

# Initiate kNN, Train the data, then test it with the date for k=3
knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)
'''
ret, result, neighbors, distance = knn.findNearest(test, k=3)

# Now we check the accuracy of the classification
# For that, compare the result with test_labels and check which are wrong
matches = (result == test_labels)
correct = np.count_nonzero(matches)
accuracy = correct * (100.0 / result.size)
print("Accuracy is = %.2f" % accuracy + "%")
'''


#image = cv2.imread('OpenCV-with-Python-master/images/numbers.jpg')
image = cv2.imread('/mnt/d/python/num3.jpeg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
        squared = makeSquare(roi)
        final = resize_to_pixel(20, squared)
        #print(final.size)
        final_array = final.reshape(-1, 400).astype(np.float32)
        ret, result, neighbours, dist = knn.findNearest(final_array, k=1)
        number = str(int(float(result[0])))
        full_number.append(number)
        

print("The number is : " + ''.join(full_number))
