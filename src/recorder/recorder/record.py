import cv2
from rclpy.node import Node 
import rclpy
import numpy as np

'''def main():
    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
    count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret==True:
            frame = cv2.flip(frame,0)

            # write the flipped frame
            out.write(frame)
            count+=1
            if cv2.waitKey(1) & count > 300:
                break
        else:
            break

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()
'''

class VideoRecorder(Node):
    def __init__(self):
        super().__init__('video_recorder')
        timerPeriod = 0.1
        self.timer = self.create_timer(timerPeriod, self.timer_callback)
        self.count = 0
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.out = cv2.VideoWriter('testing.avi', self.fourcc, 1//timerPeriod, (self.frame_width, self.frame_height))

    def fill_image(img):
        h, w = img.shape
        # Fills the area inside the tape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(img, mask, (w//2, h-1), 255)
        return img
        
    def warp_img(img, points, w, h, inv=False):
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        if inv:
            matrix = cv2.getPerspectiveTransform(pts2, pts1)
        else:
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgWarp = cv2.warpPerspective(img, matrix, (w, h))
        return imgWarp
    
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

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret==True:
            try:
                grayImg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                h, w = grayImg.shape
                blurImg = cv2.blur(grayImg, (3,3))
                cannyImg = cv2.Canny(blurImg, 0, 200)
                warpedImg = self.warp_img(cannyImg, [[30., 300.], [610., 300.], [0.,480.], [650., 480.]], h, w)
                filledImg = self.fill_image(warpedImg)
                _, histImgSmall = self.getHistogram(filledImg, display_hist=True, minPercentage=0.1, region=5)
                _, histImgLarge = self.getHistogram(filledImg, display_hist=True, minPercentage=0.1, region=1)

                imgStack = self.stackImages(0.5, ([frame, cannyImg, warpedImg], [filledImg, histImgSmall, histImgLarge]))

                self.out.write(imgStack)
                self.count+=1
                if cv2.waitKey(1) & self.count > 600:
                    exit()
            except:
                pass

def main():
    rclpy.init()

    video_recorder = VideoRecorder()

    rclpy.spin(video_recorder)

    video_recorder.destroy_node()

    rclpy.shutdown()

if __name__ == '__main__':
    main()