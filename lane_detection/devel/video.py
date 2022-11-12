# Created by Torki at 11.11.2022

""" maskwithedge.py - Short description """

import cv2
import numpy as np

cap = cv2.VideoCapture('vid_compressed.mp4')

start_frame_number = 50
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

frame_count = 0
# loop runs if capturing has been initialized
while (1):
    # reads frames from a camera

    ret, frame = cap.read()


    frame = cv2.resize(frame,(320,180),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    frame_count += 1

    # Every 20 frame
    if frame_count % 10 == 0:
        # converting BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # define range of red color in HSV
        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        # create a red HSV colour boundary and
        # threshold HSV image
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # Display an original image
        cv2.imshow('Original', frame)

        # finds edges in the input image and
        # marks them in the output map edges
        edges = cv2.Canny(frame, 255, 255)

        h, w = edges.shape

        # Fills the area inside the tape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(edges, mask, (0, 0), 255)

        # Display edges in a frame
        cv2.imshow('Edges', edges)

        # Wait for Esc key to stop
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

# Close the window
cap.release()

# De-allocate any associated memory usage
cv2.destroyAllWindows()


# # Left turn
# for col in range(0, w//2):
#     for row in range(height, 0, -1):
#         if edges[row, col] == 255:
#             break
#         edges[row, col] = 255
#
#
# # Right turn
# for col in range(0, w//2):
#     for row in range(height, 0, -1):
#         if edges[row, col] == 255:
#             break
#         edges[row, col] = 255



