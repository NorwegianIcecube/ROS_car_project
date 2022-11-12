# Created by Torki at 11.11.2022

""" maskwithedge.py - Short description """


import cv2
import numpy as np


img = cv2.imread('straight.jpg', cv2.COLOR_BGR2HSV)

# converting BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of red color in HSV
lower_red = np.array([30, 150, 50])
upper_red = np.array([255, 255, 180])

# create a red HSV colour boundary and
# threshold HSV image
mask = cv2.inRange(hsv, lower_red, upper_red)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

# Display an original image
cv2.imshow('Original', img)

# finds edges in the input image and
# marks them in the output map edges
edges = cv2.Canny(img, threshold1=255, threshold2=255, L2gradient=False)
cv2.imwrite('canny.png', edges)

h, w = edges.shape

# Fills the area inside the tape
mask = np.zeros((h + 2, w + 2), np.uint8)
cv2.floodFill(edges, mask, (0, 0), 255)


# Display edges in the img
cv2.imshow('Edges', edges)


# Wait for Esc key to stop
cv2.waitKey(0)
cv2.destroyAllWindows()

