import cv2
import numpy as np
import matplotlib.pyplot as plt


# read an image about digits from 0 to 10
# that image in shape of 2000 pixels in width and 1000 pixels in the hight
# each cell have shape of 20pixels in width and 20 pixels in the hight
img = cv2.imread('.image\digits.png',0)
#cv2.imshow('digits',img)


# crop the original image into smaller cell
# just using 50 digits for training and the other for valid and test
cells = [np.hsplit(row,100) for row in np.vsplit(img,50)]
# print(cells[0][0])
cv2.imwrite(".image\digit0.jpg",cells[0][0])
cv2.imwrite(".image\digit1.jpg",cells[6][0])
cv2.imwrite(".image\digit2.jpg",cells[11][0])
cv2.imwrite(".image\digit3.jpg",cells[16][0])
cv2.imwrite(".image\digit4.jpg",cells[21][0])


# change those cell in to matrix
matrix = np.array(cells)
#print(matrix)


# reshape that cells into an  array
array1 = matrix.reshape(-1,400)
array2= matrix.reshape(-1,200)
cv2.imwrite(".image\image1.jpg",array1)
cv2.imwrite(".image\image2.jpg",array2)


# create an train data base on the original image 
train = matrix[:,:50].reshape(-1,400).astype(np.float32)


# create an test data base on the original image 
test = matrix[:,50:100].reshape(-1,400).astype(np.float32)


# create labels of training data
# what is the function of repeat and newaxis ? 
k = np.arange(10)
train_labels = np.repeat(k,250)[:np.newaxis]


# detection digits
# 1. create an model of machine learning
# 2. then training
KNN = cv2.ml.KNearest_create()
KNN.train(train,0,train_labels)


# testing
# 1. read an image of one digits and change that image into matrix of int
# 2. change that matrix into an array with the length equal 400
# 3. then change that matrix into type of float
# 4. using function of Knearest with 5 neighbours
test_img = cv2.imread(".image\digit1.jpg",0)
test_matrix = np.array(test_img)
test = test_matrix.reshape(-1,400).astype(np.float32)
result1, result2, result3, result4 = KNN.findNearest(test,5)
print(type(result2))
result  = result2.astype(np.int)
print("That digit is: ",result)