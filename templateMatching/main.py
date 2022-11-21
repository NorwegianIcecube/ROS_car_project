import cv2
import numpy as np
from matplotlib import pyplot as plt
from stackimages import stackImages

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


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
            if sign_type == 'deploy':
                cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th), (0,255, 255), 2)
                print('DEPLOYING ANTS!!!')

            elif sign_type == 'load':
                cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th), (255, 255, 0), 2)
                print('LOADING ANTS!!!')

            else:
                pass

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