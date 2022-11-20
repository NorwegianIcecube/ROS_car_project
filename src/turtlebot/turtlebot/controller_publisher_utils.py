import cv2
import numpy as np
from collections import Counter

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def fill_image(img):
    h, w = img.shape
    # Fills the area inside the tape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(img, mask, (w//2, h-1), 255)
    return img


def warp_img(img,points,w,h,inv=False):
    
    pts1 = np.float32(points)                                   # points of starting image
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])                # points for the warp
    if inv:
        matrix = cv2.getPerspectiveTransform(pts2,pts1)
    else:
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(w,h))
    return imgWarp


def _pass(a):
    pass

'''
# Trackbar for easier testing and finding the right position for the track points
def initializeTrackbars(intialTracbarVals,wT=480, hT=240):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 550, 240)
    cv2.createTrackbar("Width Top", "Trackbars", intialTracbarVals[0],wT//2, _pass)
    cv2.createTrackbar("Height Top", "Trackbars", intialTracbarVals[1], hT, _pass)
    cv2.createTrackbar("Width Bottom", "Trackbars", intialTracbarVals[2],wT//2, _pass)
    cv2.createTrackbar("Height Bottom", "Trackbars", intialTracbarVals[3], hT, _pass)
'''
'''
def valTrackbars(wT=480, hT=240):
    widthTop = cv2.getTrackbarPos("Width Top", "Trackbars")
    heightTop = cv2.getTrackbarPos("Height Top", "Trackbars")
    widthBottom = cv2.getTrackbarPos("Width Bottom", "Trackbars")
    heightBottom = cv2.getTrackbarPos("Height Bottom", "Trackbars")
    points = np.float32([(widthTop, heightTop), (wT-widthTop, heightTop),
                      (widthBottom , heightBottom ), (wT-widthBottom, heightBottom)])
    return points
'''

def drawPoints(img,points):
    imgWarpPoints = img.copy()
    for x in range( 0,4):
        cv2.circle(imgWarpPoints,(int(points[x][0]),int(points[x][1])),15,(0,0,255),cv2.FILLED)
    return imgWarpPoints


def getHistogram(img, display_hist=False, minPercentage=0.1, region=1):
    if region == 1:
        # Looking at the whole image
        histValues = np.sum(img, axis=0)
    else:
        # Looking at the bottom of the image to find the center of the path
        histValues = np.sum(img[img.shape[0] // region:, :], axis=0)

    maxVal = np.max(histValues)
    minVal = minPercentage * maxVal                         # lower threshold for the histogram value

    indexArray = np.where(histValues >= minVal)             # sorts the indexes where the histogram values are greater
                                                            # than the lower threshold
    basePoint = int(np.average(indexArray))                 #

    if display_hist:
        imgHist = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for x, intensity in enumerate(histValues):
            # print(intensity)
            if intensity > minVal:
                color = (255, 0, 255)
            else:
                color = (0, 0, 255)
            cv2.line(imgHist, (x, img.shape[0]), (x, int(img.shape[0] - (intensity // 255 // region))), color, 1)
        cv2.circle(imgHist, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)       # plot the basepoint
        return basePoint, imgHist

    return basePoint


def gray_hist_avg(hist):
    
    hist = cv2.cvtColor(hist, cv2.COLOR_BGR2GRAY)
    
    temp = [Counter(col) for col in zip(*hist)]
    white_count_col = [[i, temp[i][max(temp[i].keys())]] for i in range(len(temp))]
    avg = 0
    num_whites = 0
    for tpl in white_count_col:
        avg += tpl[0] * tpl[1]
        num_whites += tpl[1]
        
    avg = avg // num_whites

    return avg


def pipeline(img, points, turn):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    img_blur = cv2.blur(img_gray, (3,3))
    img_canny = cv2.Canny(img_blur, 0, 200)
   
    #valTrackbars(IMAGE_WIDTH, IMAGE_HEIGHT)
    img_warp = warp_img(img_canny, points, h, w)
    img_fill = img_warp.copy()
    fill_image(img_fill)
    _1, lanePositionHist = getHistogram(img_fill, display_hist=True, minPercentage=0.2, region=6)
    _2, fullHist = getHistogram(img_fill, display_hist=True, minPercentage=0.9, region=1)
    
    avg = _1#gray_hist_avg(fullHist)
    mid = _2#gray_hist_avg(lanePositionHist)
    
    treshold = 10

    cv2.line(fullHist, (mid, fullHist.shape[0]), (mid, fullHist.shape[1]), (0, 255, 255), 2)
    cv2.line(fullHist, (avg, 0), (avg, fullHist.shape[0]), (255, 0, 0), 2)
    cv2.line(fullHist, (avg + treshold, 0), (avg + treshold, fullHist.shape[0]), (0, 255, 0), 2)
    cv2.line(fullHist, (avg - treshold, 0), (avg - treshold, fullHist.shape[0]), (0, 255, 0), 2)
    
    
    
    
    if avg < mid - treshold:
        #turn -= 0.05 #steers right
        turn = (avg-mid)/1000
    
    elif avg > mid + treshold:
        #turn += 0.05 #steers left
        turn = (avg-mid)/1000
    else:
        turn = 0.0

    img_stack = stackImages(0.6, ([img, img_canny, img_warp],
                                    [img_fill, lanePositionHist, fullHist]))


    
        
    return turn, img_stack
    

# Test the functions if the module is run
if __name__ == '__main__':
    print("Hello World!")
