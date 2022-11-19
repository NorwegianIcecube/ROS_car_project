import cv2
import numpy as np
import utils
import stackimages

TESTING = True

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

def onBot():
    pass

def onTest():
    vidcap = cv2.VideoCapture('src/turtlebot3_antmobile/turtlebot3_antmobile/drive.avi')
    success, image = vidcap.read()

    initialTrackbarVals = [30, 300, 0, 480] #width top, height top, width bottom, height bottom. Slider trackbar
    utils.initializeTrackbars(initialTrackbarVals, IMAGE_WIDTH, IMAGE_HEIGHT)

    frameCounter = 0
    while success:
        frameCounter += 1
        if vidcap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        _, img = vidcap.read()  # GET THE IMAGE
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))  # RESIZE
        pipeline(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q To exit
            break
    vidcap.release()
    cv2.destroyAllWindows()



def pipeline(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img_gray, (3, 3), 0)

    h, w = img.shape
    points = utils.valTrackbars(IMAGE_WIDTH, IMAGE_HEIGHT)
    imgWarp = utils.warp_img(img, points, w, h)
    imgWarpPoints = utils.drawPoints(img, points)

    sigm = np.std(imgWarp)
    mean = np.mean(imgWarp)
    lower = int(max(0, (mean - sigm)))
    upper = int(min(255, (mean + sigm)))

    #box blur to deNoise
    boxBlur = cv2.blur(imgWarp, (5,5))
    edges = cv2.Canny(boxBlur, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    img_filled = edges.copy()
    utils.fill_image(img_filled)

    vehicle_pos_in_lane, imgHist = utils.getHistogram(imgWarp, display_hist=True, minPercentage=0.5, region=4)
    curvature, btm_img_hist = utils.getHistogram(imgWarp, display_hist=True, minPercentage=0.9)

    print(curvature-vehicle_pos_in_lane)

    imgStacked = stackimages.stackImages(0.7, ([imgWarpPoints, boxBlur, edges],
                                            [img_filled, imgHist, btm_img_hist]))
    cv2.imshow('ImageStack', imgStacked)



def main():
    if TESTING:
        onTest()
    else:
        onBot()

if __name__ == '__main__':
    main()