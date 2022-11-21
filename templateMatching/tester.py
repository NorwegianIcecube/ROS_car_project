import sys
import cv2
import numpy

img = cv2.imread(sys.argv[1])

img2 = img[:,:,2]
img2 = img2 - cv2.erode(img2, None)
template = cv2.imread(sys.argv[2])[:,:,2]
template = template - cv2.erode(template, None)

ccnorm = cv2.matchTemplate(img2, template, cv2.TM_CCORR_NORMED)
print(ccnorm.max())
loc = numpy.where(ccnorm == ccnorm.max())
threshold = 0.4
th, tw = template.shape[:2]
for pt in zip(*loc[::-1]):
    if ccnorm[pt[::-1]] < threshold:
        continue
    cv2.rectangle(img, pt, (pt[0] + tw, pt[1] + th),
            (0, 0, 255), 2)

cv2.imwrite(sys.argv[2], img)