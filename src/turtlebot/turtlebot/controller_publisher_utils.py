import cv2
import numpy as np
from collections import Counter
import math

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

def threshold_match(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([166, 60, 70])
    upper_red = np.array([179, 200, 255])
    masked_red = cv2.inRange(hsv, lower_red, upper_red)

    # Threshold of blue in HSV space
    lower_blue = np.array([109, 50, 105])
    upper_blue = np.array([120, 210, 185])
    masked_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    nr_red = np.argwhere(masked_red == 255)
    nr_blue = np.argwhere(masked_blue == 255)

    threshold = 1500
    print(len(nr_red))
    if len(nr_red) > threshold:
        return "red"
    else:
        return "blue"

def template_match(_img, template):
    img = _img#.copy()
    img2 = img[:, :, 2]
    img2 = img2 - cv2.erode(img2, None)

    template = template[:, :, 2]
    template = template - cv2.erode(template, None)

    ccnorm = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)

    ccnorm.max()
    loc = np.where(ccnorm == ccnorm.max())
    threshold = 0.4
    th, tw = template.shape[:2]
    for pt in zip(*loc[::-1]):
        if ccnorm[pt[::-1]] > threshold:
            cropped_img = img.copy()
            cropped_img = cropped_img[(pt[1]):(pt[1] + th), pt[0]:(pt[0] + tw)]

            command = threshold_match(cropped_img)
    
            cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th), (0, 0, 255), 2)
            return command #, img



def feature_matching(template, scene, treshold):

    sift = cv2.SIFT_create()
    
    kp1, des1 = sift.detectAndCompute(template,None)
    kp2, des2 = sift.detectAndCompute(scene,None)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1,des2,k=2)
    
    # david lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(template,kp1,scene,kp2,good,None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    if len(good) > treshold:
        return True, img3, len(good)

    else:
        return False, None, 0


def pipeline(img, points, turn, load_ants_template, deploy_ants_template):
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img_gray.shape
    img_blur = cv2.blur(img_gray, (3,3))
    img_canny = cv2.Canny(img_blur, 50, 200, L2gradient=True)
   
    #valTrackbars(IMAGE_WIDTH, IMAGE_HEIGHT)
    img_warp = warp_img(img_canny, points, h, w)
    img_fill = img_warp.copy()
    fill_image(img_fill)
    _1, lanePositionHist = getHistogram(img_fill, display_hist=True, minPercentage=0.2, region=6)
    _2, fullHist = getHistogram(img_fill, display_hist=True, minPercentage=0.9, region=1)
    
    avg = _1#gray_hist_avg(fullHist)
    mid = _2#gray_hist_avg(lanePositionHist)
    
    treshold = 35

    cv2.line(fullHist, (mid, fullHist.shape[0]), (mid, fullHist.shape[1]), (0, 255, 255), 2)
    cv2.line(fullHist, (avg, 0), (avg, fullHist.shape[0]), (255, 0, 0), 2)
    cv2.line(fullHist, (avg + treshold, 0), (avg + treshold, fullHist.shape[0]), (0, 255, 0), 2)
    cv2.line(fullHist, (avg - treshold, 0), (avg - treshold, fullHist.shape[0]), (0, 255, 0), 2)
    
    
    if avg < mid - treshold or avg > mid + treshold:
        #turn -= 0.05 #steers right
        turn = math.sqrt(abs((avg-mid)/2000))
        if avg < mid:
            turn = -turn
    else:
        turn = 0.0


    cv2.line(fullHist, (avg, fullHist.shape[0]), (avg-int(turn*1000), fullHist.shape[1]), (255, 255, 0), 2)

    speed = 0.1

    stop = False
    pause = False
    img_original = img
    #command = template_match(img, load_ants_template)
    #command = template_match(img, deploy_ants_template)

    feature_matches = 35

    cmd, matched_img, n_matches = feature_matching(load_ants_template, img, feature_matches)
    cmd2, matched_img2, n_matches2 = feature_matching(deploy_ants_template, img, feature_matches)

    if n_matches >= n_matches2 and cmd == True:
        pause = cmd
        img_fill = matched_img

    elif n_matches <= n_matches2 and cmd2 == True:
        stop = cmd2
        img_fill = matched_img2
    

    img_stack = stackImages(0.6, ([img, img_canny, img_warp],
                                    [img_fill, lanePositionHist, fullHist]))

    #if command == "red":
    #    stop = True
    #elif command == "blue":
    #    pause = True
        
    return turn, img_stack, speed, stop, pause
    

# Test the functions if the module is run
if __name__ == '__main__':
    print("Hello World!")
