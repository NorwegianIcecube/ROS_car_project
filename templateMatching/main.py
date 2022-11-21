import cv2
import numpy as np
from matplotlib import pyplot as plt
from stackimages import stackImages

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


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
        print("DEPLOY ANTS!!")
        cv2.imshow("red", masked_red)
    else:
        print("LOAD ANTS!!")
        cv2.imshow("blue", masked_blue)


def template_match(_img, template, sign_type):
    img = _img.copy()
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
            cv2.imshow("cropped", cropped_img)

            threshold_match(cropped_img)

    return img2, img


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # img_original = cv2.imread('img.jpg')

    load_ants_template = cv2.imread('load_ants.jpg')
    load_ants_template = cv2.resize(load_ants_template, (200, 200))

    deploy_ants_template = cv2.imread('deploy_ants.jpg')
    deploy_ants_template = cv2.resize(deploy_ants_template, (200, 200))

    while True:
        _, img_original = cap.read(0)  # GET THE IMAGE

        # Apply template Matching
        bw_res_img1, res_img1 = template_match(img_original, load_ants_template, 'deploy')
        bw_res_img2, res_img2 = template_match(img_original, deploy_ants_template, 'load')

        imgStacked = stackImages(0.5, ([img_original, bw_res_img2], [res_img1, res_img2]))
        cv2.imshow('ImageStack', imgStacked)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press q To exit
            break

    # cap.release()
    cv2.destroyAllWindows()