import cv2
import numpy as np
import utils
import stackimages

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# TODO
#   - Endre størrelsen på bildene når vi skal kjøre realtime
#   - Om minne er et problem kan


def getLaneCurve(img):
    ### PIPELINE ###
    # STEP 1 - Edge Detection and Fill Lane #
    imgThres = utils.thresholding(img)  # Make image black and white

    edges = utils.edge_detection(img)

    cv2.imshow('Warped', edges)

    # imgThres = utils.fill_image(edges)

    # STEP 2 - Warp Image to Bird View #
    h, w, c = img.shape
    points = utils.valTrackbars(IMAGE_WIDTH, IMAGE_HEIGHT)
    imgWarp = utils.warp_img(img, points, w, h)
    imgWarp = utils.thresholding(imgWarp)
    imgWarpPoints = utils.drawPoints(img, points)

    # STEP 3 - Make Histogram #
    vehicle_pos_in_lane, imgHist = utils.getHistogram(imgWarp, display_hist=True, minPercentage=0.5, region=4)
    curvature, btm_img_hist = utils.getHistogram(imgWarp, display_hist=True, minPercentage=0.9)

    # Gives the amount need to turn
    print(curvature-vehicle_pos_in_lane)

    # STEP 4 - Averages out curve to handle noise better#
    # TODO trengs den?

    # SHOW IMAGES #
    imgStacked = stackimages.stackImages(0.7, ([img, imgWarpPoints, imgWarp],
                                         [imgThres, imgHist, btm_img_hist]))
    cv2.imshow('ImageStack', imgStacked)

    return imgThres


def pipeline(img):
    # Convert to graycsale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img = cv2.GaussianBlur(img_gray, (3, 3), 0)

    # Warp Image
    h, w = img.shape
    points = utils.valTrackbars(IMAGE_WIDTH, IMAGE_HEIGHT)
    imgWarp = utils.warp_img(img, points, w, h)
    imgWarpPoints = utils.drawPoints(img, points)

    # Canny Edge Detection
    sigma = np.std(imgWarp)
    mean = np.mean(imgWarp)
    lower = int(max(0, (mean - sigma)))
    upper = int(min(255, (mean + sigma)))
    edges = cv2.Canny(imgWarp, lower, upper)

    # Dilate edges to increase robustness
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Fill Lane
    img_filled = edges.copy()
    utils.fill_image(img_filled)

    # Histogram
    vehicle_pos_in_lane, imgHist = utils.getHistogram(imgWarp, display_hist=True, minPercentage=0.5, region=4)
    curvature, btm_img_hist = utils.getHistogram(imgWarp, display_hist=True, minPercentage=0.9)

    # Gives the amount need to turn
    print(curvature-vehicle_pos_in_lane)

    # STEP 4 - Averages out curve to handle noise better#
    # TODO trengs den?

    # SHOW IMAGES #
    imgStacked = stackimages.stackImages(0.7, ([imgWarpPoints, imgWarp, edges],
                                               [img_filled, imgHist, btm_img_hist]))
    cv2.imshow('ImageStack', imgStacked)


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)

    intialTracbarVals = [170, 120, 70, 430]  # width top, height top, width bottom, height bottom. Slider trackbar
    utils.initializeTrackbars(intialTracbarVals, IMAGE_WIDTH, IMAGE_HEIGHT)

    frameCounter = 0
    while True:
        # Restarts the video if end is reached
        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        _, img = cap.read()  # GET THE IMAGE
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))  # RESIZE
        pipeline(img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q To exit
            break

    cap.release()
    cv2.destroyAllWindows()
